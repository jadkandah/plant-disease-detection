'''
MultiModalResNet50 not so efficent but were keeping it just incase
the other file is overfitting
working fine not the best model
'''
import os
from pathlib import Path

import torch
import torch.nn as nn
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
ROOT = Path("/home/jad/plant-disease-detection/")
DATASET_PATH = ROOT / "jordan_dataset"
METADATA_CSV = DATASET_PATH / "metadata_weather.csv"
from pathlib import Path

MODEL_PATH = ROOT / "src" / "dissdetector" / "resnet50_multimodal_plant_disease_improved.pth"
print("MODEL_PATH =", MODEL_PATH)
print("Exists?", MODEL_PATH.exists())
BATCH_SIZE = 8
IMAGE_SIZE = 512

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {DEVICE}")

FEATURE_COLS = [
    "temp_c",
    "humidity_pct",
    "wind_m_s",
    "precip_mm",
    "soil_moisture_pct",
]

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

test_transforms = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ToTensorV2(),
])

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_leaf_classes(split_dir):
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
    df = pd.read_csv(csv_path)
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["image_rel_path"] = df["image_rel_path"].astype(str)
    return df


def compute_feature_stats(train_df):
    means = train_df[FEATURE_COLS].mean()
    stds = train_df[FEATURE_COLS].std().fillna(1.0).replace(0, 1.0)
    return means, stds


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
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.split_name = split_name
        self.class_to_idx = class_to_idx
        self.feature_means = feature_means
        self.feature_stds = feature_stds

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
            raise RuntimeError(f"No valid samples found under: {self.root_dir}")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, rel_path, target = self.samples[index]

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        img_np = np.array(img)
        try:
            out = self.transform(image=img_np)
            img_tensor = out["image"].contiguous()
        except Exception:
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
            feat_vals.append(float((val - self.feature_means[col]) / std))

        feat_tensor = torch.tensor(feat_vals, dtype=torch.float32)
        return img_tensor, feat_tensor, target


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return (
            torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.empty(0, len(FEATURE_COLS)),
            torch.empty(0, dtype=torch.long),
        )
    return default_collate(batch)


class MultiModalResNet50(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()

        backbone = models.resnet50(weights=None)
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


def update_confusion_matrix(conf_mat, labels, preds, num_classes):
    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    for t, p in zip(labels, preds):
        conf_mat[t, p] += 1
    return conf_mat


def compute_metrics_from_confusion(conf_mat):
    total = conf_mat.sum()
    correct = np.trace(conf_mat)
    acc = correct / total if total > 0 else float("nan")

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


def main():
    split_dirs, classes, class_to_idx = build_shared_mapping(DATASET_PATH)
    metadata_df = load_metadata(METADATA_CSV)

    train_df = metadata_df[metadata_df["image_rel_path"].str.startswith("train/")].copy()
    feature_means, feature_stds = compute_feature_stats(train_df)

    test_dataset = MultiModalPlantDataset(
        root_dir=split_dirs["test"],
        transform=test_transforms,
        image_size=IMAGE_SIZE,
        split_name="test",
        class_to_idx=class_to_idx,
        metadata_df=metadata_df,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=safe_collate,
    )

    model = MultiModalResNet50(num_classes=len(class_to_idx), num_features=len(FEATURE_COLS))
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    conf_mat = np.zeros((len(class_to_idx), len(class_to_idx)), dtype=np.int64)
    seen = 0

    with torch.no_grad():
        for images, features, labels in tqdm(test_loader, desc="Test"):
            if images.numel() == 0:
                continue

            images = images.to(DEVICE)
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images, features)
            preds = torch.argmax(outputs, dim=1)

            conf_mat = update_confusion_matrix(conf_mat, labels, preds, len(class_to_idx))
            seen += images.size(0)

    test_acc, test_miou = compute_metrics_from_confusion(conf_mat)
    print(f"\nTest samples evaluated: {seen}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test mIoU: {test_miou:.4f}")


if __name__ == "__main__":
    main()