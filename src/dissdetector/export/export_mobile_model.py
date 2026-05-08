"""Export an image-only PyTorch checkpoint to ONNX for offline mobile inference.

This utility intentionally supports only image-only classifiers. The online
multimodal model still belongs on the backend because it requires metadata
features and backend-only SAM preprocessing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dissdetector.models.model_factory import create_model, is_multimodal_model


DEFAULT_OUTPUT_DIR = ROOT / "mobile_models"
DEFAULT_ONNX_NAME = "offline_model.onnx"
DEFAULT_MANIFEST_NAME = "offline_model_manifest.json"
DEFAULT_DATASET_ROOT = ROOT / "jordan_dataset"
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ExportConfig:
    checkpoint: Path
    output_dir: Path
    onnx_name: str
    manifest_name: str
    manifest: Path | None
    model_name: str | None
    num_classes: int | None
    image_size: int | None
    dataset_root: Path | None
    dataset_variant: str | None
    class_mapping_json: Path | None
    opset: int
    device: str
    validate: bool


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(
        description="Export an image-only .pth checkpoint to ONNX for React Native.",
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the .pth state_dict.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--onnx-name", default=DEFAULT_ONNX_NAME)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--manifest", type=Path, help="Optional saved training manifest.")
    parser.add_argument("--model-name", help="Override model_name when no manifest exists.")
    parser.add_argument("--num-classes", type=int, help="Override class count when no mapping is available.")
    parser.add_argument("--image-size", type=int, help="Override image size when no manifest exists.")
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        type=Path,
        help="Dataset root used to rebuild class_mapping when no manifest exists.",
    )
    parser.add_argument(
        "--dataset-variant",
        default=None,
        help="Dataset variant for class mapping rebuild: original or background_removed.",
    )
    parser.add_argument("--class-mapping-json", type=Path, help="JSON object mapping class label to index.")
    parser.add_argument("--opset", default=17, type=int)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    return ExportConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        onnx_name=args.onnx_name,
        manifest_name=args.manifest_name,
        manifest=args.manifest,
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        dataset_root=args.dataset_root,
        dataset_variant=args.dataset_variant,
        class_mapping_json=args.class_mapping_json,
        opset=args.opset,
        device=args.device,
        validate=not args.skip_validation,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def find_manifest(checkpoint: Path, explicit_manifest: Path | None) -> tuple[dict[str, Any], Path | None]:
    candidates: list[Path] = []
    if explicit_manifest:
        candidates.append(explicit_manifest)
    candidates.extend(
        [
            checkpoint.with_name(f"{checkpoint.stem}_manifest.json"),
            checkpoint.with_suffix(".json"),
            checkpoint.parent / "manifest.json",
        ]
    )

    for candidate in candidates:
        if candidate and candidate.is_file():
            return load_json(candidate), candidate

    return {}, None


def infer_model_name(checkpoint: Path) -> str | None:
    name = checkpoint.stem.lower()
    if "multimodal" in name or "metadata_only" in name or "late_fusion" in name:
        return "multimodal_resnet50"
    if "mobilenet_v3_small" in name:
        return "mobilenet_v3_small"
    if "mobilenet_v3_large" in name:
        return "mobilenet_v3_large"
    if "efficientnet_b0" in name:
        return "efficientnet_b0"
    if "resnet50" in name:
        return "image_only_resnet50"
    return None


def infer_image_size(checkpoint: Path) -> int | None:
    matches = [int(value) for value in re.findall(r"(?:^|_)(224|256|512)(?:_|$)", checkpoint.stem)]
    return matches[-1] if matches else None


def infer_dataset_variant(checkpoint: Path) -> str | None:
    name = checkpoint.stem.lower()
    if "background_removed" in name:
        return "background_removed"
    if "full_data_set" in name or "original" in name:
        return "original"
    return None


def normalize_class_mapping(mapping: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for label, index in mapping.items():
        normalized[str(label)] = int(index)
    values = sorted(normalized.values())
    expected = list(range(len(normalized)))
    if values != expected:
        raise ValueError(
            "class_mapping must contain contiguous zero-based indices. "
            f"Got indices {values[:10]}... for {len(values)} classes."
        )
    return dict(sorted(normalized.items(), key=lambda item: item[1]))


def resolve_split_dirs(base_dir: Path, dataset_variant: str | None) -> dict[str, Path]:
    variant = (dataset_variant or "background_removed").strip().lower()
    if variant == "background_removed":
        root = base_dir
    elif variant == "original":
        root = base_dir / "Original"
    else:
        raise ValueError("dataset_variant must be 'original' or 'background_removed'.")

    split_dirs = {split: root / split for split in ("train", "val", "test")}
    missing = [f"{split}: {path}" for split, path in split_dirs.items() if not path.is_dir()]
    if missing:
        raise RuntimeError(f"Missing dataset split directories: {', '.join(missing)}")
    return split_dirs


def class_mapping_from_dataset(dataset_root: Path, dataset_variant: str | None) -> dict[str, int]:
    classes: set[str] = set()
    for split_dir in resolve_split_dirs(dataset_root, dataset_variant).values():
        for crop_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for disease_dir in sorted(path for path in crop_dir.iterdir() if path.is_dir()):
                classes.add(f"{crop_dir.name}___{disease_dir.name}")

    if not classes:
        raise RuntimeError(f"No class folders found under dataset root: {dataset_root}")

    class_to_idx = {label: index for index, label in enumerate(sorted(classes))}
    return normalize_class_mapping(class_to_idx)


def resolve_class_mapping(config: ExportConfig, manifest: dict[str, Any]) -> dict[str, int] | None:
    if "class_mapping" in manifest:
        return normalize_class_mapping(manifest["class_mapping"])

    if config.class_mapping_json:
        return normalize_class_mapping(load_json(config.class_mapping_json))

    if config.dataset_root and config.dataset_root.is_dir():
        return class_mapping_from_dataset(
            config.dataset_root,
            dataset_variant=config.dataset_variant or infer_dataset_variant(config.checkpoint),
        )

    return None


def extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                checkpoint_obj = value
                break

    if not isinstance(checkpoint_obj, dict):
        raise TypeError("Checkpoint must be a state_dict or a dict containing state_dict/model_state_dict.")

    state_dict = {}
    for key, value in checkpoint_obj.items():
        normalized_key = key.removeprefix("module.")
        state_dict[normalized_key] = value
    return state_dict


def build_manifest_payload(
    *,
    config: ExportConfig,
    source_manifest_path: Path | None,
    model_name: str,
    image_size: int,
    class_mapping: dict[str, int],
    onnx_path: Path,
) -> dict[str, Any]:
    index_to_class = {str(index): label for label, index in class_mapping.items()}
    return {
        "format": "onnx",
        "runtime": "onnxruntime-react-native",
        "offline_scope": "image_only",
        "model_name": model_name,
        "num_classes": len(class_mapping),
        "class_mapping": class_mapping,
        "index_to_class": index_to_class,
        "image_size": image_size,
        "input_name": "image",
        "output_name": "logits",
        "input_shape": [1, 3, image_size, image_size],
        "preprocessing": {
            "color_space": "RGB",
            "resize": [image_size, image_size],
            "normalize_mean": NORM_MEAN,
            "normalize_std": NORM_STD,
            "tensor_layout": "NCHW",
            "value_range_before_normalize": [0.0, 1.0],
        },
        "source_checkpoint": str(config.checkpoint),
        "source_manifest": str(source_manifest_path) if source_manifest_path else None,
        "onnx_file": onnx_path.name,
        "dataset_variant": config.dataset_variant or infer_dataset_variant(config.checkpoint),
        "notes": [
            "React Native must load this ONNX file with onnxruntime-react-native.",
            "Do not run multimodal metadata models offline on mobile.",
            "Do not run SAM background removal in the frontend.",
        ],
    }


def validate_onnx_export(model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: Path) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping ONNX runtime validation.")
        return

    with torch.no_grad():
        torch_logits = model(dummy_input).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(["logits"], {"image": dummy_input.detach().cpu().numpy()})[0]
    max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
    print(f"ONNX validation max_abs_diff={max_abs_diff:.8f}")
    if max_abs_diff > 1e-4:
        raise RuntimeError(f"ONNX output differs from PyTorch output by {max_abs_diff:.8f}")


def export_to_onnx(config: ExportConfig) -> tuple[Path, Path]:
    if not config.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")

    manifest, manifest_path = find_manifest(config.checkpoint, config.manifest)
    model_name = config.model_name or manifest.get("model_name") or infer_model_name(config.checkpoint)
    if not model_name:
        raise ValueError("Could not infer model_name. Pass --model-name or provide a manifest.")

    if is_multimodal_model(model_name):
        raise ValueError(
            f"Refusing to export multimodal model '{model_name}' for offline mobile inference. "
            "Export an image-only checkpoint instead."
        )

    image_size = config.image_size or manifest.get("image_size") or infer_image_size(config.checkpoint)
    if not image_size:
        raise ValueError("Could not infer image_size. Pass --image-size or provide a manifest.")
    image_size = int(image_size)

    class_mapping = resolve_class_mapping(config, manifest)
    if class_mapping is None:
        if config.num_classes is None:
            raise ValueError(
                "No class_mapping available. Provide a manifest, --class-mapping-json, "
                "--dataset-root, or --num-classes."
            )
        class_mapping = {f"class_{i}": i for i in range(config.num_classes)}
        print("Warning: generated placeholder class labels because only --num-classes was provided.")

    num_classes = config.num_classes or len(class_mapping)
    if int(num_classes) != len(class_mapping):
        raise ValueError(f"num_classes={num_classes} does not match class_mapping size={len(class_mapping)}")

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(config.device)

    model = create_model(model_name=model_name, num_classes=len(class_mapping)).to(device)
    checkpoint_obj = torch.load(config.checkpoint, map_location=device)
    state_dict = extract_state_dict(checkpoint_obj)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = config.output_dir / config.onnx_name
    output_manifest_path = config.output_dir / config.manifest_name

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=config.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )

    if config.validate:
        validate_onnx_export(model, dummy_input, onnx_path)

    payload = build_manifest_payload(
        config=config,
        source_manifest_path=manifest_path,
        model_name=model_name,
        image_size=image_size,
        class_mapping=class_mapping,
        onnx_path=onnx_path,
    )
    output_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return onnx_path, output_manifest_path


def main() -> None:
    onnx_path, manifest_path = export_to_onnx(parse_args())
    print(f"Wrote ONNX model: {onnx_path}")
    print(f"Wrote mobile manifest: {manifest_path}")


if __name__ == "__main__":
    main()
