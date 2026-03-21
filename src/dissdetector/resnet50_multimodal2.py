#resnet50 multimodal plant disease detection with weather/soil data
import os
import sys
import time
import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision.models as models

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import cv2
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# Configuration
# =========================
ROOT = Path("/Users/sanadmadani/plant-disease-detection/plant-disease-detection")
DATASET_PATH = ROOT / "jordan_dataset2"
METADATA_CSV = DATASET_PATH / "metadata_weather.csv"

BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
IMAGE_SIZE = 384
PATIENCE = 5
SEED = 42

MODEL_OUTPUT_PATH = ROOT / "src" / "dissdetector" / "resnet50_multimodal_plant_disease_improved.pth"

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

# =========================
# Seed
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# =========================
# Weather / soil feature columns
# =========================
FEATURE_COLS = [
    "temp_c",
    "humidity_pct",
    "wind_m_s",
    "precip_mm",
    "soil_moisture_pct",
]

# =========================
# Transforms
# =========================
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transforms = A.Compose([
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        translate_percent=0.04,
        scale=(0.95, 1.05),
        rotate=15,
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT
    ),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

val_test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# Helpers
# =========================
def list_leaf_classes(split_dir):
    classes = set()
    for parent in sorted(os.listdir(split_dir)):
        parent_path = split_dir / parent
        if not parent_path.is_dir():
            continue
        for leaf in sorted(os.listdir(parent_path)):
            leaf_path = split_dir / parent / leaf
            if leaf_path.is_dir():
                classes.add(f"{parent}___{leaf}")
    return classes


def build_shared_mapping(base_dir):
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


def load_metadata(csv_path):
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


def compute_feature_stats(train_df):
    means = train_df[FEATURE_COLS].mean()
    stds = train_df[FEATURE_COLS].std()
    stds = stds.fillna(1.0).replace(0, 1.0)
    return means, stds


def compute_class_weights_from_samples(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, _, cls_idx in samples:
        counts[cls_idx] += 1.0

    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


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
    for c in range(conf_mat.shape[0]):
        tp = conf_mat[c, c]
        fp = conf_mat[:, c].sum() - tp
        fn = conf_mat[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)

    miou = float(np.mean(ious)) if len(ious) > 0 else float("nan")
    return acc, miou


# =========================
# Dataset
# =========================
class MultiModalPlantDataset(Dataset):
    def __init__(
        self,
        root_dir,
        transform,
        image_size,
        split_name,
        class_to_idx,
        metadata_df,
        feature_means,
        feature_stds,
        log_limit=50,
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

        split_prefix = f"{split_name}/"
        metadata_df = metadata_df[metadata_df["image_rel_path"].str.startswith(split_prefix)].copy()
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

                for fname in os.listdir(leaf_path):
                    fpath = leaf_path / fname
                    if not (fpath.is_file() and fpath.suffix.lower() in IMG_EXTS):
                        continue

                    rel_path = f"{split_name}/{parent_name}/{leaf_name}/{fname}"
                    if rel_path not in self.meta_map:
                        continue

                    samples.append((str(fpath), rel_path, cls_idx))

        if len(samples) == 0:
            raise RuntimeError(f"No valid multimodal samples found under: {self.root_dir}")

        self.samples = samples
        self.classes = sorted(self.class_to_idx.keys())

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
        path, rel_path, target = self.samples[index]

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

        try:
            out = self.transform(image=img_np)
            img_tensor = out["image"].contiguous()
        except Exception as e:
            self._log_bad(path, f"transform error ({e})")
            return None

        if not (
            img_tensor.ndim == 3
            and img_tensor.shape[0] == 3
            and img_tensor.shape[1] == self.image_size
            and img_tensor.shape[2] == self.image_size
        ):
            self._log_bad(path, f"bad tensor shape {tuple(img_tensor.shape)}")
            return None

        row = self.meta_map[rel_path]
        feat_vals = []

        for col in FEATURE_COLS:
            val = row.get(col, np.nan)
            if pd.isna(val):
                val = self.feature_means[col]

            std = self.feature_stds[col]
            if pd.isna(std) or std == 0:
                std = 1.0

            val = (val - self.feature_means[col]) / std
            feat_vals.append(float(val))

        feat_tensor = torch.tensor(feat_vals, dtype=torch.float32)

        return img_tensor, feat_tensor, target


# =========================
# Safe collate
# =========================
def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return (
            torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.empty(0, len(FEATURE_COLS)),
            torch.empty(0, dtype=torch.long),
        )
    return default_collate(batch)


# =========================
# Data loading
# =========================
def load_data(base_dir, metadata_csv):
    split_dirs, classes, class_to_idx = build_shared_mapping(base_dir)

    metadata_df = load_metadata(metadata_csv)
    train_df = metadata_df[metadata_df["image_rel_path"].str.startswith("train/")].copy()
    if len(train_df) == 0:
        raise RuntimeError("No training metadata rows found in metadata CSV.")

    feature_means, feature_stds = compute_feature_stats(train_df)

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = MultiModalPlantDataset(
            root_dir=split_dirs[split],
            transform=train_transforms if split == "train" else val_test_transforms,
            image_size=IMAGE_SIZE,
            split_name=split,
            class_to_idx=class_to_idx,
            metadata_df=metadata_df,
            feature_means=feature_means,
            feature_stds=feature_stds,
            log_limit=50,
        )

    print("\n--- Shared Model Label Mapping (Text to Integer) ---")
    print({c: class_to_idx[c] for c in sorted(class_to_idx)})
    print("----------------------------------------------------")

    dataloaders = {}
    for split in ["train", "val", "test"]:
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=False,
            collate_fn=safe_collate,
        )

    dataset_sizes = {split: len(datasets[split]) for split in datasets}

    print("\nDataset sizes:")
    for split in ["train", "val", "test"]:
        print(
            f"  {split}: {dataset_sizes[split]} samples "
            f"(~{(dataset_sizes[split] + BATCH_SIZE - 1)//BATCH_SIZE} batches @ batch_size={BATCH_SIZE})"
        )

    return dataloaders, datasets, dataset_sizes, class_to_idx


# =========================
# Multimodal Model
# =========================
class MultiModalResNet50(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all first
        for p in backbone.parameters():
            p.requires_grad = False

        # Unfreeze only layer4
        for p in backbone.layer4.parameters():
            p.requires_grad = True

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, features):
        img_vec = self.image_backbone(images)
        feat_vec = self.feature_mlp(features)
        x = torch.cat([img_vec, feat_vec], dim=1)
        return self.classifier(x)


def load_model(num_classes):
    model = MultiModalResNet50(num_classes=num_classes, num_features=len(FEATURE_COLS))
    model.to(DEVICE)
    print(f"\nLoaded improved multimodal ResNet-50 model for {num_classes} classes.")
    return model


# =========================
# Train / Eval
# =========================
def run_epoch(model, dataloader, criterion, optimizer=None, num_classes=1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    seen = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if is_train else None

    for images, features, labels in tqdm(dataloader, desc="train" if is_train else "eval"):
        if images.numel() == 0:
            continue

        images = images.to(DEVICE)
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    outputs = model(images, features)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images, features)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        seen += bs
        conf_mat = update_confusion_matrix(conf_mat, labels, preds, num_classes)

    epoch_loss = running_loss / seen if seen > 0 else float("nan")
    epoch_acc, epoch_miou = compute_metrics_from_confusion(conf_mat)

    return epoch_loss, epoch_acc, epoch_miou


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, patience, num_classes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        train_loss, train_acc, train_miou = run_epoch(
            model, dataloaders["train"], criterion, optimizer=optimizer, num_classes=num_classes
        )
        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f} mIoU: {train_miou:.4f}")

        val_loss, val_acc, val_miou = run_epoch(
            model, dataloaders["val"], criterion, optimizer=None, num_classes=num_classes
        )
        print(f"val   Loss: {val_loss:.4f} Acc: {val_acc:.4f} mIoU: {val_miou:.4f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print("New best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s")
    print(f"Best val Acc: {best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model


# =========================
# Main
# =========================
if __name__ == "__main__":
    if not DATASET_PATH.is_dir():
        print(f"ERROR: Data directory not found at {DATASET_PATH}")
        sys.exit(1)

    if not METADATA_CSV.is_file():
        print(f"ERROR: Metadata CSV not found at {METADATA_CSV}")
        sys.exit(1)

    dataloaders, datasets, dataset_sizes, class_to_idx = load_data(DATASET_PATH, METADATA_CSV)
    num_classes = len(class_to_idx)

    model_ft = load_model(num_classes)

    class_weights = compute_class_weights_from_samples(datasets["train"].samples, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    backbone_params = list(model_ft.image_backbone.layer4.parameters())
    head_params = list(model_ft.feature_mlp.parameters()) + list(model_ft.classifier.parameters())

    optimizer_ft = optim.Adam([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params, "lr": 1e-3},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft,
        mode="min",
        factor=0.5,
        patience=2
    )

    print("\nStarting improved multimodal training...")
    model_ft = train_model(
        model_ft,
        dataloaders,
        criterion,
        optimizer_ft,
        scheduler,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        num_classes=num_classes
    )

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_ft.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nModel saved successfully to {MODEL_OUTPUT_PATH}")

    print("\n--- Final Test Set Evaluation ---")
    test_loss, test_acc, test_miou = run_epoch(
        model_ft,
        dataloaders["test"],
        criterion,
        optimizer=None,
        num_classes=num_classes
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} (on {dataset_sizes['test']} samples)")
    print(f"Test mIoU: {test_miou:.4f}")
