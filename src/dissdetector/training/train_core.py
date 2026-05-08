import copy
import os
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from src.dissdetector.config.runtime import (
    DATASET_PATH,
    DEFAULT_SEED,
    DEFAULT_SELECTION_METRIC,
    build_dataloader_generator,
    resolve_split_dirs,
    selection_metric_mode,
    validate_selection_metric,
)
from src.dissdetector.training.early_stopping import EarlyStopping
from src.dissdetector.training.metrics import compute_confusion_matrix, compute_miou_from_confmat


ROOT = Path("/home/jad/plant-disease-detection")

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def cuda_amp_enabled() -> bool:
    disabled = os.getenv("DISSDETECTOR_DISABLE_AMP", "").strip().lower()
    return DEVICE.type == "cuda" and disabled not in _TRUTHY_ENV_VALUES


def _save_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_model_wts,
    best_acc: float,
    best_val_loss: float,
    early_stopper: EarlyStopping,
    selection_metric: str,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "next_epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "best_model_wts": best_model_wts,
            "best_acc": best_acc,
            "best_val_loss": best_val_loss,
            "early_stopper": {
                "best_value": early_stopper.best_value,
                "counter": early_stopper.counter,
                "should_stop": early_stopper.should_stop,
            },
            "selection_metric": selection_metric,
        },
        checkpoint_path,
    )


def _load_training_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    early_stopper: EarlyStopping,
):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler.is_enabled() and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    stopper_state = checkpoint.get("early_stopper", {})
    early_stopper.best_value = stopper_state.get("best_value")
    early_stopper.counter = stopper_state.get("counter", 0)
    early_stopper.should_stop = stopper_state.get("should_stop", False)

    return (
        int(checkpoint.get("next_epoch", 0)),
        checkpoint.get("best_model_wts", copy.deepcopy(model.state_dict())),
        float(checkpoint.get("best_acc", 0.0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


def build_transforms(image_size: int):
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.0625,
            scale=(0.6, 1.0),
            rotate=25,
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        # A.CoarseDropout(max_holes=8, max_height=64, max_width=64, min_holes=1, fill_value=0, p=0.5), # it gave some issues with some images and they were skipped in training
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        A.RandomShadow(p=0.3),
        A.GaussianBlur(p=0.2),
        ToTensorV2()

    ])

    val_test_transforms = A.Compose([
        A.Resize(image_size, image_size),
        A.CenterCrop(image_size, image_size, p=1.0),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])

    return train_transforms, val_test_transforms


def list_leaf_classes(split_dir: Path):
    classes = set()
    for parent in sorted(os.listdir(split_dir)):
        parent_path = split_dir / parent
        if not parent_path.is_dir():
            continue
        for leaf in sorted(os.listdir(parent_path)):
            leaf_path = parent_path / leaf
            if leaf_path.is_dir():
                classes.add(f"{parent}___{leaf}")
    return classes


def build_shared_mapping(base_dir: Path, dataset_variant: str | None = None):
    split_dirs = resolve_split_dirs(base_dir, dataset_variant=dataset_variant)

    all_classes = set()
    for split, sd in split_dirs.items():
        if not sd.is_dir():
            raise RuntimeError(f"Missing split directory for {split}: {sd}")
        all_classes |= list_leaf_classes(sd)

    classes = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return split_dirs, classes, class_to_idx


class LeafClassAlbumentationsDataset(Dataset):
    def __init__(self, root_dir: Path, transform, image_size: int, split_name: str, class_to_idx: dict, log_limit: int = 20):
        self.root_dir = str(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.split_name = split_name
        self.class_to_idx = class_to_idx

        self._bad_logged = set()
        self._bad_count = 0
        self._log_limit = log_limit

        samples = []
        for parent_name in sorted(os.listdir(self.root_dir)):
            parent_path = os.path.join(self.root_dir, parent_name)
            if not os.path.isdir(parent_path):
                continue

            for leaf_name in sorted(os.listdir(parent_path)):
                leaf_path = os.path.join(parent_path, leaf_name)
                if not os.path.isdir(leaf_path):
                    continue

                cls = f"{parent_name}___{leaf_name}"
                if cls not in self.class_to_idx:
                    continue

                cls_idx = self.class_to_idx[cls]

                for fname in sorted(os.listdir(leaf_path)):
                    fpath = os.path.join(leaf_path, fname)
                    if os.path.isfile(fpath) and Path(fpath).suffix.lower() in IMG_EXTS:
                        samples.append((fpath, cls_idx))

        if len(samples) == 0:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _log_bad(self, path, msg):
        if self._bad_count < self._log_limit and path not in self._bad_logged:
            print(f"[{self.split_name}] Skipping file due to {msg}: {path}")
            self._bad_logged.add(path)
            self._bad_count += 1

    def __getitem__(self, index):
        path, target = self.samples[index]

        try:
            img = Image.open(path)
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError, Exception) as e:
            self._log_bad(path, f"read/convert error ({e})")
            return None

        img_np = np.array(img)

        try:
            out = self.transform(image=img_np)
            img_tensor = out["image"].contiguous()
        except Exception as e:
            self._log_bad(path, f"transform error ({e})")
            return None

        if not (
            img_tensor.ndim == 3 and
            img_tensor.shape[0] == 3 and
            img_tensor.shape[1] == self.image_size and
            img_tensor.shape[2] == self.image_size
        ):
            self._log_bad(path, f"bad tensor shape {tuple(img_tensor.shape)}")
            return None

        return img_tensor, target


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return default_collate(batch)


def load_data(
    base_dir: Path,
    image_size: int,
    batch_size: int,
    dataset_variant: str | None = None,
    seed: int = DEFAULT_SEED,
):
    train_transforms, val_test_transforms = build_transforms(image_size)
    split_dirs, classes, class_to_idx = build_shared_mapping(
        base_dir=base_dir,
        dataset_variant=dataset_variant,
    )
    dataloader_generator = build_dataloader_generator(seed)

    datasets = {
        split: LeafClassAlbumentationsDataset(
            root_dir=split_dirs[split],
            transform=train_transforms if split == "train" else val_test_transforms,
            image_size=image_size,
            split_name=split,
            class_to_idx=class_to_idx,
        )
        for split in ["train", "val", "test"]
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=safe_collate,
            generator=dataloader_generator if split == "train" else None,
        )
        for split in ["train", "val", "test"]
    }

    dataset_sizes = {split: len(datasets[split]) for split in datasets}
    return dataloaders, dataset_sizes, class_to_idx


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    num_classes,
    num_epochs=5,
    patience=5,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    checkpoint_path: Path | None = None,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float("inf")
    selection_metric = validate_selection_metric(selection_metric)

    use_amp = cuda_amp_enabled()
    if DEVICE.type == "cuda":
        print(f"CUDA AMP enabled: {use_amp}")

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=0.0,
        mode=selection_metric_mode(selection_metric),
    )

    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            (
                start_epoch,
                best_model_wts,
                best_acc,
                best_val_loss,
            ) = _load_training_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                early_stopper=early_stopper,
            )
            print(f"Resuming from checkpoint: {checkpoint_path} (epoch {start_epoch + 1}/{num_epochs})")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            seen = 0
            confmat_epoch = torch.zeros((num_classes, num_classes), dtype=torch.long)

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                if inputs.numel() == 0:
                    continue

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                bs = inputs.size(0)
                running_loss += loss.item() * bs
                running_corrects += torch.sum(preds == labels).item()
                seen += bs
                confmat_epoch += compute_confusion_matrix(preds, labels, num_classes)

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / seen if seen > 0 else float("nan")
            epoch_acc = running_corrects / seen if seen > 0 else float("nan")
            epoch_miou = compute_miou_from_confmat(confmat_epoch)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} mIoU: {epoch_miou:.4f}")

            if phase == "val" and seen > 0:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss

                selected_value = epoch_acc if selection_metric == "best_val_acc" else epoch_loss
                if (
                    selection_metric == "best_val_acc" and epoch_acc >= best_acc
                ) or (
                    selection_metric == "best_val_loss" and epoch_loss <= best_val_loss
                ):
                    best_model_wts = copy.deepcopy(model.state_dict())

                early_stopper.step(selected_value)

                if checkpoint_path is not None:
                    _save_training_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_model_wts=best_model_wts,
                        best_acc=best_acc,
                        best_val_loss=best_val_loss,
                        early_stopper=early_stopper,
                        selection_metric=selection_metric,
                    )
                    print(f"Checkpoint saved: {checkpoint_path}")

        print()

        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break

    time_elapsed = time.time() - since
    print(f"Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f"Best val Acc: {best_acc:.4f}")
    print(f"Best val Loss: {best_val_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, best_acc, best_val_loss


def evaluate_model(model, dataloader, num_classes):
    model.eval()

    running_corrects = 0
    seen = 0
    confmat = torch.zeros((num_classes, num_classes), dtype=torch.long)
    use_amp = cuda_amp_enabled()

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Test phase"):
            if inputs.numel() == 0:
                continue

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels).item()
            seen += inputs.size(0)
            confmat += compute_confusion_matrix(preds, labels, num_classes)

    accuracy = running_corrects / seen if seen > 0 else float("nan")
    miou = compute_miou_from_confmat(confmat)

    return {
        "accuracy": accuracy,
        "miou": miou,
    }
