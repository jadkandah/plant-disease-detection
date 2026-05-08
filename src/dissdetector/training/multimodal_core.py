import copy
import os
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
    normalize_dataset_variant,
    resolve_split_dirs,
    selection_metric_mode,
    validate_selection_metric,
)
from src.dissdetector.models.model_factory import create_model
from src.dissdetector.training.early_stopping import EarlyStopping


ROOT = Path("/home/jad/plant-disease-detection")
METADATA_CSV = DATASET_PATH / "metadata_weather.csv"

TRAIN_SPLIT_DIRNAME = "train"
FEATURE_COLS = [
    "temp_c",
    "humidity_pct",
    "wind_m_s",
    "precip_mm",
    "soil_moisture_pct",
]

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METADATA_FEATURES = FEATURE_COLS


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_transforms = A.Compose([
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.90, 1.0),
            ratio=(0.95, 1.05),
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent=0.03,
            scale=(0.97, 1.03),
            rotate=10,
            p=0.4,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
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


def load_metadata(csv_path: Path):
    if not csv_path.is_file():
        raise RuntimeError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"image_rel_path", *FEATURE_COLS}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required metadata columns: {missing}")

    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["image_rel_path"] = df["image_rel_path"].astype(str)
    return df


def validate_metadata_variant(metadata_csv: Path, dataset_variant: str | None) -> None:
    variant = normalize_dataset_variant(dataset_variant)

    if variant == "background_removed":
        return

    if variant == "original":
        raise RuntimeError(
            "dataset_variant='original' uses raw images under jordan_dataset/Original, "
            "but no compatible raw-image metadata CSV is configured. "
            f"{metadata_csv} matches the active background-removed train/val/test dataset."
        )


def compute_feature_stats(train_df: pd.DataFrame):
    means = train_df[FEATURE_COLS].mean()
    stds = train_df[FEATURE_COLS].std()
    stds = stds.fillna(1.0).replace(0, 1.0)
    return means, stds


def compute_class_weights_from_samples(samples, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, _, cls_idx in samples:
        counts[cls_idx] += 1.0

    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = np.clip(weights, 0.3, 5.0)

    return torch.tensor(weights, dtype=torch.float32)


def update_confusion_matrix(conf_mat, labels, preds, num_classes):
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    for t, p in zip(labels, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf_mat[t, p] += 1
    return conf_mat


def compute_metrics_from_confusion(conf_mat):
    total = conf_mat.sum()
    correct = np.trace(conf_mat)
    acc = (correct / total) if total > 0 else float("nan")

    ious = []
    f1s = []

    for c in range(conf_mat.shape[0]):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp

        denom_iou = tp + fp + fn
        if denom_iou > 0:
            ious.append(tp / denom_iou)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    miou = float(np.mean(ious)) if len(ious) > 0 else float("nan")
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else float("nan")

    return acc, miou, macro_f1


def build_feature_tensor(row, feature_means, feature_stds):
    feature_values = []

    for col in FEATURE_COLS:
        value = row.get(col, np.nan)
        if pd.isna(value):
            value = feature_means[col]

        std = feature_stds[col]
        if pd.isna(std) or std == 0:
            std = 1.0

        normalized_value = (value - feature_means[col]) / std
        feature_values.append(float(normalized_value))

    return torch.tensor(feature_values, dtype=torch.float32)


class MultiModalPlantDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        transform,
        image_size: int,
        split_name: str,
        class_to_idx: dict,
        metadata_df: pd.DataFrame,
        feature_means,
        feature_stds,
        log_limit: int = 50,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.split_name = split_name
        self.class_to_idx = class_to_idx
        self.feature_means = feature_means
        self.feature_stds = feature_stds

        self._bad_logged = set()
        self._bad_count = 0
        self._log_limit = log_limit

        metadata_prefix = "train/" if split_name == "train" else f"{split_name}/"
        metadata_df = metadata_df[metadata_df["image_rel_path"].str.startswith(metadata_prefix)].copy()
        self.meta_map = metadata_df.set_index("image_rel_path").to_dict(orient="index")

        samples = []
        for parent_name in sorted(os.listdir(self.root_dir)):
            parent_path = self.root_dir / parent_name
            if not parent_path.is_dir():
                continue

            for leaf_name in sorted(os.listdir(parent_path)):
                leaf_path = parent_path / leaf_name
                if not leaf_path.is_dir():
                    continue

                cls = f"{parent_name}___{leaf_name}"
                if cls not in self.class_to_idx:
                    continue

                cls_idx = self.class_to_idx[cls]

                for fname in sorted(os.listdir(leaf_path)):
                    fpath = leaf_path / fname
                    if not (fpath.is_file() and fpath.suffix.lower() in IMG_EXTS):
                        continue

                    metadata_rel_path = f"{metadata_prefix}{parent_name}/{leaf_name}/{fname}"
                    if metadata_rel_path not in self.meta_map:
                        continue

                    samples.append((str(fpath), metadata_rel_path, cls_idx))

        if len(samples) == 0:
            raise RuntimeError(f"No valid multimodal samples found under: {self.root_dir}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _log_bad(self, path, msg):
        if self._bad_count < self._log_limit and path not in self._bad_logged:
            print(f"[{self.split_name}] Skipping file due to {msg}: {path}")
            self._bad_logged.add(path)
            self._bad_count += 1

    def __getitem__(self, index):
        path, metadata_rel_path, target = self.samples[index]

        try:
            img = Image.open(path).convert("RGB")
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

        row = self.meta_map[metadata_rel_path]
        feat_tensor = build_feature_tensor(row, self.feature_means, self.feature_stds)
        return img_tensor, feat_tensor, target


class MetadataOnlyPlantDataset(Dataset):
    _placeholder_image = torch.zeros(3, 1, 1, dtype=torch.float32)

    def __init__(
        self,
        split_name: str,
        class_to_idx: dict,
        metadata_df: pd.DataFrame,
        feature_means,
        feature_stds,
    ):
        self.split_name = split_name
        self.class_to_idx = class_to_idx
        self.feature_means = feature_means
        self.feature_stds = feature_stds

        metadata_prefix = "train/" if split_name == "train" else f"{split_name}/"
        split_df = metadata_df[metadata_df["image_rel_path"].str.startswith(metadata_prefix)].copy()

        samples = []
        for _, row in split_df.sort_values(by="image_rel_path").iterrows():
            parts = row["image_rel_path"].split("/")
            if len(parts) < 4:
                continue

            cls = f"{parts[1]}___{parts[2]}"
            if cls not in class_to_idx:
                continue

            samples.append((row.to_dict(), row["image_rel_path"], class_to_idx[cls]))

        if len(samples) == 0:
            raise RuntimeError(f"No valid metadata-only samples found for split: {split_name}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row, _, target = self.samples[index]
        feature_tensor = build_feature_tensor(row, self.feature_means, self.feature_stds)
        return self._placeholder_image, feature_tensor, target


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return (
            torch.empty(0, 3, 1, 1),
            torch.empty(0, len(FEATURE_COLS)),
            torch.empty(0, dtype=torch.long),
        )
    return default_collate(batch)


def load_multimodal_data(
    base_dir: Path,
    metadata_csv: Path,
    image_size: int,
    batch_size: int,
    dataset_variant: str | None = None,
    seed: int = DEFAULT_SEED,
    model_name: str | None = None,
):
    train_transforms, val_test_transforms = build_transforms(image_size)
    split_dirs, classes, class_to_idx = build_shared_mapping(
        base_dir=base_dir,
        dataset_variant=dataset_variant,
    )

    validate_metadata_variant(metadata_csv, dataset_variant)
    metadata_df = load_metadata(metadata_csv)
    train_df = metadata_df[metadata_df["image_rel_path"].str.startswith("train/")].copy()
    if len(train_df) == 0:
        raise RuntimeError("No training metadata rows found in metadata CSV.")

    feature_means, feature_stds = compute_feature_stats(train_df)
    dataset_cls = MetadataOnlyPlantDataset if model_name == "metadata_only_mlp" else MultiModalPlantDataset
    dataloader_generator = build_dataloader_generator(seed)

    datasets = {
        split: dataset_cls(
            **(
                {
                    "root_dir": split_dirs[split],
                    "transform": train_transforms if split == "train" else val_test_transforms,
                    "image_size": image_size,
                    "split_name": split,
                    "class_to_idx": class_to_idx,
                    "metadata_df": metadata_df,
                    "feature_means": feature_means,
                    "feature_stds": feature_stds,
                    "log_limit": 50,
                }
                if dataset_cls is MultiModalPlantDataset
                else {
                    "split_name": split,
                    "class_to_idx": class_to_idx,
                    "metadata_df": metadata_df,
                    "feature_means": feature_means,
                    "feature_stds": feature_stds,
                }
            )
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
            drop_last=(split == "train"),
            generator=dataloader_generator if split == "train" else None,
        )
        for split in ["train", "val", "test"]
    }

    dataset_sizes = {split: len(datasets[split]) for split in datasets}
    return dataloaders, datasets, dataset_sizes, class_to_idx


def _append_param_group(param_groups, module, lr: float):
    if module is None:
        return

    params = [p for p in module.parameters() if p.requires_grad]
    if params:
        param_groups.append({"params": params, "lr": lr})


def build_optimizer_phase1(model, model_name: str, learning_rate: float):
    # metadata-only model does not have an image backbone
    if model_name == "metadata_only_mlp":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    param_groups = []

    # Keep the same rough ratio your current early-fusion model was using:
    # backbone layer4 lower LR, new heads higher LR
    _append_param_group(param_groups, getattr(model.image_backbone, "layer4", None), learning_rate * 0.1)
    _append_param_group(param_groups, getattr(model, "image_proj", None), learning_rate * 0.5)
    _append_param_group(param_groups, getattr(model, "image_classifier", None), learning_rate * 0.5)
    _append_param_group(param_groups, getattr(model, "feature_mlp", None), learning_rate * 0.5)
    _append_param_group(param_groups, getattr(model, "feature_classifier", None), learning_rate * 0.5)
    _append_param_group(param_groups, getattr(model, "fusion", None), learning_rate * 0.5)
    _append_param_group(param_groups, getattr(model, "fusion_head", None), learning_rate * 0.5)
    _append_param_group(param_groups, getattr(model, "classifier", None), learning_rate * 0.5)

    return optim.AdamW(param_groups, weight_decay=1e-4)


def build_optimizer_phase2(model, model_name: str, learning_rate: float):
    if model_name == "metadata_only_mlp":
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    param_groups = []

    _append_param_group(param_groups, getattr(model.image_backbone, "layer3", None), learning_rate * 0.03)
    _append_param_group(param_groups, getattr(model.image_backbone, "layer4", None), learning_rate * 0.05)
    _append_param_group(param_groups, getattr(model, "image_proj", None), learning_rate * 0.3)
    _append_param_group(param_groups, getattr(model, "image_classifier", None), learning_rate * 0.3)
    _append_param_group(param_groups, getattr(model, "feature_mlp", None), learning_rate * 0.3)
    _append_param_group(param_groups, getattr(model, "feature_classifier", None), learning_rate * 0.3)
    _append_param_group(param_groups, getattr(model, "fusion", None), learning_rate * 0.3)
    _append_param_group(param_groups, getattr(model, "fusion_head", None), learning_rate * 0.3)
    _append_param_group(param_groups, getattr(model, "classifier", None), learning_rate * 0.3)

    return optim.AdamW(param_groups, weight_decay=1e-4)

def forward_multimodal_model(model, model_name: str, images, features):
    if model_name == "metadata_only_mlp":
        return model(features)

    if model_name in {"multimodal_resnet50", "late_fusion_multimodal_resnet50"}:
        return model(images, features)

    raise ValueError(f"Unsupported multimodal pipeline model_name: {model_name}")

def run_multimodal_epoch(model, model_name: str, dataloader, criterion, optimizer=None, num_classes: int = 1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    seen = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if is_train else None

    for images, features, labels in tqdm(dataloader, desc="train" if is_train else "eval"):
        if labels.numel() == 0:
            continue

        # BatchNorm safety
        if is_train and labels.size(0) < 2:
            continue

        images = images.to(DEVICE, non_blocking=True)
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    outputs = forward_multimodal_model(
                        model=model,
                        model_name=model_name,
                        images=images,
                        features=features,
                    )
                    loss = criterion(outputs, labels)
            else:
                outputs = forward_multimodal_model(
                    model=model,
                    model_name=model_name,
                    images=images,
                    features=features,
                )
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        seen += bs
        conf_mat = update_confusion_matrix(conf_mat, labels, preds, num_classes)

    epoch_loss = running_loss / seen if seen > 0 else float("nan")
    epoch_acc, epoch_miou, epoch_f1 = compute_metrics_from_confusion(conf_mat)

    return epoch_loss, epoch_acc, epoch_miou, epoch_f1


def build_plateau_scheduler(optimizer):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )


def _unfreeze_layer3_if_available(model) -> bool:
    if not (hasattr(model, "image_backbone") and hasattr(model.image_backbone, "layer3")):
        return False

    for p in model.image_backbone.layer3.parameters():
        p.requires_grad = True

    return True


def _save_multimodal_checkpoint(
    checkpoint_path: Path,
    model,
    model_name: str,
    optimizer,
    scheduler,
    epoch: int,
    best_model_wts,
    best_val_loss: float,
    best_val_acc: float,
    early_stopper: EarlyStopping,
    selection_metric: str,
    unfreeze_epoch: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "next_epoch": epoch + 1,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_model_wts": best_model_wts,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "early_stopper": {
                "best_value": early_stopper.best_value,
                "counter": early_stopper.counter,
                "should_stop": early_stopper.should_stop,
            },
            "selection_metric": selection_metric,
            "unfreeze_epoch": unfreeze_epoch,
        },
        checkpoint_path,
    )


def _load_multimodal_checkpoint(
    checkpoint,
    model,
    optimizer,
    scheduler,
    early_stopper: EarlyStopping,
):
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    stopper_state = checkpoint.get("early_stopper", {})
    early_stopper.best_value = stopper_state.get("best_value")
    early_stopper.counter = stopper_state.get("counter", 0)
    early_stopper.should_stop = stopper_state.get("should_stop", False)

    return (
        checkpoint.get("best_model_wts", copy.deepcopy(model.state_dict())),
        float(checkpoint.get("best_val_loss", float("inf"))),
        float(checkpoint.get("best_val_acc", -1.0)),
    )


def train_multimodal_model(
    model,
    model_name: str,
    dataloaders,
    criterion,
    learning_rate: float,
    num_epochs: int,
    patience: int,
    num_classes: int,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    checkpoint_path: Path | None = None,
):
    since = time.time()
    selection_metric = validate_selection_metric(selection_metric)
    unfreeze_epoch = 4

    checkpoint = None
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            start_epoch = int(checkpoint.get("next_epoch", 0))
            print(f"Resuming from checkpoint: {checkpoint_path} (epoch {start_epoch + 1}/{num_epochs})")

    if start_epoch > unfreeze_epoch and _unfreeze_layer3_if_available(model):
        print("Layer3 already unfrozen from checkpoint.")

    if start_epoch > unfreeze_epoch:
        optimizer = build_optimizer_phase2(
            model=model,
            model_name=model_name,
            learning_rate=learning_rate
        )
    else:
        optimizer = build_optimizer_phase1(
            model=model,
            model_name=model_name,
            learning_rate=learning_rate
        )

    scheduler = build_plateau_scheduler(optimizer)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = -1.0
    early_stopper = EarlyStopping(
        patience=patience,
        min_delta=0.0,
        mode=selection_metric_mode(selection_metric),
    )

    if checkpoint is not None:
        best_model_wts, best_val_loss, best_val_acc = _load_multimodal_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
        )

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 12)

        if epoch == unfreeze_epoch and _unfreeze_layer3_if_available(model):
            print("Unfreezing layer3...")

            optimizer = build_optimizer_phase2(
                model=model,
                model_name=model_name,
                learning_rate=learning_rate
            )

            scheduler = build_plateau_scheduler(optimizer)

        train_loss, train_acc, train_miou, train_f1 = run_multimodal_epoch(
            model=model,
            model_name=model_name,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            num_classes=num_classes
        )
        print(
            f"train Loss: {train_loss:.4f} | "
            f"Acc: {train_acc:.4f} | "
            f"mIoU: {train_miou:.4f} | "
            f"Macro-F1: {train_f1:.4f}"
        )

        val_loss, val_acc, val_miou, val_f1 = run_multimodal_epoch(
            model=model,
            model_name=model_name,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=None,
            num_classes=num_classes
        )
        print(
            f"val   Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"mIoU: {val_miou:.4f} | "
            f"Macro-F1: {val_f1:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        selected_value = val_acc if selection_metric == "best_val_acc" else val_loss
        if (
            selection_metric == "best_val_acc" and val_acc >= best_val_acc
        ) or (
            selection_metric == "best_val_loss" and val_loss <= best_val_loss
        ):
            best_model_wts = copy.deepcopy(model.state_dict())
            print("New best model saved.")

        early_stopper.step(selected_value)

        if checkpoint_path is not None:
            _save_multimodal_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                model_name=model_name,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_model_wts=best_model_wts,
                best_val_loss=best_val_loss,
                best_val_acc=best_val_acc,
                early_stopper=early_stopper,
                selection_metric=selection_metric,
                unfreeze_epoch=unfreeze_epoch,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        if early_stopper.should_stop:
            print("Early stopping triggered.")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f"Best val Loss: {best_val_loss:.4f}")
    print(f"Best val Acc : {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, best_val_acc, best_val_loss

def evaluate_multimodal_model(model, model_name: str, dataloader, criterion, num_classes: int):
    test_loss, test_acc, test_miou, test_f1 = run_multimodal_epoch(
        model=model,
        model_name=model_name,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=None,
        num_classes=num_classes
    )

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "miou": test_miou,
        "macro_f1": test_f1,
    }

def create_multimodal_training_objects(model_name: str, num_classes: int):
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        num_features=len(FEATURE_COLS)
    ).to(DEVICE)

    return model

def get_multimodal_loss(datasets, num_classes: int):
    class_weights = compute_class_weights_from_samples(
        datasets["train"].samples,
        num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    return criterion
