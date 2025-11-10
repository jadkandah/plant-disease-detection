import os
import sys
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# Configuration
# =========================
ROOT = Path("/home/jad/plant-disease-detection/plant-disease-detection")
DATASET_PATH = ROOT / "jordan_dataset" / "images"

# Safer defaults for laptop GPUs; you can raise later
BATCH_SIZE = 8           # try 4 if VRAM is tight
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
IMAGE_SIZE = 384         # try 256 for even lower memory

MODEL_OUTPUT_PATH = ROOT / "src" / "dissdetector" / "resnet_50_plant_disease.pth"

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

torch.backends.cudnn.benchmark = True  # ok for fixed image size

# =========================
# Transforms (Albumentations v2)
# =========================
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# IMPORTANT: p=1.0 to guarantee fixed size every time
train_transforms = A.Compose([
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=25, p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

val_test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

# =========================
# Dataset (leaf classes) with shared mapping
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_leaf_classes(split_dir: Path):
    """Return sorted set of 'Parent___Leaf' class names present under split_dir."""
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

def build_shared_mapping(base_dir: Path):
    """Union classes across train/val/test and build a single shared mapping."""
    all_classes = set()
    split_dirs = {}
    for split in ["train", "val", "test"]:
        sd = base_dir / split
        if not sd.is_dir():
            raise RuntimeError(f"Missing split directory: {sd}")
        split_dirs[split] = sd
        all_classes |= list_leaf_classes(sd)
    classes = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return split_dirs, classes, class_to_idx

class LeafClassAlbumentationsDataset(Dataset):
    def __init__(self, root_dir: Path, transform: A.Compose | None, image_size: int,
                 split_name: str, class_to_idx: dict[str, int], log_limit: int = 50):
        self.root_dir = str(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.split_name = split_name
        self.class_to_idx = class_to_idx  # SHARED mapping across splits

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
                    # Shouldn't happen, but keep safe
                    continue
                cls_idx = self.class_to_idx[cls]

                for fname in os.listdir(leaf_path):
                    fpath = os.path.join(leaf_path, fname)
                    if os.path.isfile(fpath) and Path(fpath).suffix.lower() in IMG_EXTS:
                        samples.append((fpath, cls_idx))

        if len(samples) == 0:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        self.samples = samples
        self.classes = sorted(self.class_to_idx.keys())  # same across splits

    def __len__(self):
        return len(self.samples)

    def _log_bad(self, path, msg):
        if self._bad_count < self._log_limit and path not in self._bad_logged:
            print(f"[{self.split_name}] Skipping file due to {msg}: {path}")
            self._bad_logged.add(path)
            self._bad_count += 1
        elif self._bad_count == self._log_limit:
            print(f"[{self.split_name}] Further bad-file messages suppressed...")
            self._bad_count += 1

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path)
        except (UnidentifiedImageError, OSError) as e:
            self._log_bad(path, f"read error ({e})")
            return None

        try:
            img = img.convert("RGB")
        except Exception as e:
            self._log_bad(path, f"convert RGB error ({e})")
            return None

        img_np = np.array(img)

        if self.transform:
            try:
                out = self.transform(image=img_np)
                img_tensor = out["image"].contiguous()
            except Exception as e:
                self._log_bad(path, f"transform error ({e})")
                return None
        else:
            img_tensor = transforms.ToTensor()(img).contiguous()

        if not (img_tensor.ndim == 3 and img_tensor.shape[0] == 3 and
                img_tensor.shape[1] == self.image_size and img_tensor.shape[2] == self.image_size):
            self._log_bad(path, f"bad tensor shape {tuple(img_tensor.shape)}")
            return None

        return img_tensor, target

# =========================
# Safe collate
# =========================
from torch.utils.data._utils.collate import default_collate

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE), torch.empty(0, dtype=torch.long)
    return default_collate(batch)

# =========================
# Data loading
# =========================
def load_data(base_dir: Path):
    split_dirs, classes, class_to_idx = build_shared_mapping(base_dir)

    datasets = {
        split: LeafClassAlbumentationsDataset(
            root_dir=split_dirs[split],
            transform=train_transforms if split == "train" else val_test_transforms,
            image_size=IMAGE_SIZE,
            split_name=split,
            class_to_idx=class_to_idx,
            log_limit=50,
        ) for split in ["train", "val", "test"]
    }

    # Report shared mapping from TRAIN viewpoint for convenience
    print("\n--- Shared Model Label Mapping (Text to Integer) ---")
    print({c: class_to_idx[c] for c in sorted(class_to_idx)})
    print("----------------------------------------------------")

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=safe_collate
        )
        for split in ["train", "val", "test"]
    }

    dataset_sizes = {split: len(datasets[split]) for split in datasets}

    print("\nDataset sizes:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {dataset_sizes[split]} images "
              f"(~{(dataset_sizes[split] + BATCH_SIZE - 1)//BATCH_SIZE} batches @ batch_size={BATCH_SIZE})")

    return dataloaders, dataset_sizes, class_to_idx

# =========================
# Model
# =========================
def load_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(DEVICE)
    print(f"\nLoaded ResNet-50 model with final layer adapted for {num_classes} classes.")
    return model

# =========================
# Train (with AMP)
# =========================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == "cuda"))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            seen = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                if inputs.numel() == 0:
                    continue

                inputs = inputs.to(DEVICE, non_blocking=False)
                labels = labels.to(DEVICE, non_blocking=False)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with torch.autocast(device_type='cuda', enabled=(DEVICE.type == "cuda")):
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

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / seen if seen > 0 else float('nan')
            epoch_acc = running_corrects / seen if seen > 0 else float('nan')

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and seen > 0 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# =========================
# Main
# =========================
if __name__ == "__main__":
    if not DATASET_PATH.is_dir():
        print(f"ERROR: Data directory not found at {DATASET_PATH}. Check ROOT.")
        sys.exit(1)

    dataloaders, dataset_sizes, class_to_idx = load_data(DATASET_PATH)
    num_classes = len(class_to_idx)

    model_ft = load_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print("\nStarting Training...")
    model_ft = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=NUM_EPOCHS
    )

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_ft.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nModel saved successfully to {MODEL_OUTPUT_PATH}")

    print("\n--- Final Test Set Evaluation ---")
    model_ft.eval()
    running_corrects = 0
    seen = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["test"], desc="Test phase"):
            if inputs.numel() == 0:
                continue
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            with torch.autocast(device_type='cuda', enabled=(DEVICE.type == "cuda")):
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            seen += inputs.size(0)

    test_acc = (running_corrects / seen) if seen > 0 else float('nan')
    print(f"Test Accuracy: {test_acc:.4f} (on {seen} images)")
