from pathlib import Path
import random

import numpy as np
import torch


ROOT = Path("/home/jad/plant-disease-detection")
DATASET_PATH = ROOT / "jordan_dataset"

DEFAULT_SEED = 42
DEFAULT_DATASET_VARIANT = "background_removed"
DEFAULT_PATIENCE = 5
DEFAULT_SELECTION_METRIC = "best_val_acc"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

_SPLITS = ("train", "val", "test")
_DATASET_VARIANTS = ("background_removed", "original")
_SELECTION_MODES = {
    "best_val_acc": "max",
    "best_val_loss": "min",
}


def normalize_dataset_variant(dataset_variant: str | None) -> str:
    return (dataset_variant or DEFAULT_DATASET_VARIANT).strip().lower()


def _validate_split_dirs(split_dirs: dict[str, Path], dataset_variant: str) -> None:
    missing = [f"{split}: {path}" for split, path in split_dirs.items() if not path.is_dir()]
    if missing:
        raise RuntimeError(
            f"Dataset variant '{dataset_variant}' requires these split directories, "
            f"but they were not found: {', '.join(missing)}"
        )


def resolve_split_dirs(base_dir: Path, dataset_variant: str | None = None) -> dict[str, Path]:
    variant = normalize_dataset_variant(dataset_variant)

    if variant == "background_removed":
        split_dirs = {split: base_dir / split for split in _SPLITS}
        _validate_split_dirs(split_dirs, variant)
        return split_dirs

    if variant == "original":
        original_root = base_dir / "Original"
        split_dirs = {split: original_root / split for split in _SPLITS}
        _validate_split_dirs(split_dirs, variant)
        return split_dirs

    raise ValueError(
        f"Unsupported dataset_variant: {dataset_variant}. "
        f"Expected one of: {', '.join(_DATASET_VARIANTS)}"
    )


def validate_selection_metric(selection_metric: str | None) -> str:
    metric = (selection_metric or DEFAULT_SELECTION_METRIC).strip()
    if metric not in _SELECTION_MODES:
        raise ValueError(
            f"Unsupported selection_metric: {metric}. "
            f"Expected one of: {', '.join(sorted(_SELECTION_MODES))}"
        )
    return metric


def selection_metric_mode(selection_metric: str | None) -> str:
    metric = validate_selection_metric(selection_metric)
    return _SELECTION_MODES[metric]


def build_dataloader_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
