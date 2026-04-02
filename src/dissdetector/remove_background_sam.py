import os
import cv2
import numpy as np
from pathlib import Path
import torch

from mobile_sam import sam_model_registry, SamPredictor

# =========================
# CONFIG
# =========================
ROOT = Path("/home/jad/plant-disease-detection")

INPUT_DIR = ROOT / "jordan_dataset" / "train"
OUTPUT_DIR = ROOT / "jordan_dataset" / "train_images_background_removed"

CHECKPOINT_PATH = ROOT / "mobile_sam.pt"
MODEL_TYPE = "vit_t"

# AUTO GPU DETECTION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_PADDING = 15

# =========================
# UTILS
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model():
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
    sam.to(device=DEVICE)
    sam.eval()

    predictor = SamPredictor(sam)
    return predictor


def get_center_prompt_box(image):
    h, w = image.shape[:2]

    return np.array([
        int(w * 0.10),
        int(h * 0.15),
        int(w * 0.90),
        int(h * 0.85)
    ])


def choose_best_mask(masks, scores):
    best_idx = None
    best_value = -1

    for i, (mask, score) in enumerate(zip(masks, scores)):
        area = mask.sum()
        if area == 0:
            continue

        quality = float(score) - 0.000001 * area

        if quality > best_value:
            best_value = quality
            best_idx = i

    if best_idx is None:
        return None

    return masks[best_idx].astype(np.uint8) * 255


def clean_mask(mask):
    kernel_small = np.ones((5, 5), np.uint8)
    kernel_big = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
    return clean


def crop_from_mask(image, mask, padding=10):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, image.shape[1])
    y2 = min(y + h + padding, image.shape[0])

    return image[y1:y2, x1:x2]


def apply_mask_white_bg(image, mask):
    background = np.full_like(image, 255)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask_3ch == 255, image, background)


# =========================
# 🔥 FIXED PATH LOGIC
# =========================
def get_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel_path = input_path.relative_to(input_root)
    parts = rel_path.parts

    # EXPECTED:
    # train/Plant/Disease/image.jpg
    if len(parts) >= 3:
        plant = parts[0]
        disease = parts[1]
    elif len(parts) == 2:
        plant = parts[0]
        disease = "unknown"
    else:
        plant = "unknown"
        disease = "unknown"

    filename = Path(parts[-1]).with_suffix(".png").name

    return output_root / plant / disease / filename


# =========================
# PROCESS IMAGE
# =========================
def process_image(image_path, predictor):
    image_path = Path(image_path)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[ERROR] Cannot read {image_path}")
        return False

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    box = get_center_prompt_box(image_bgr)

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    best_mask = choose_best_mask(masks, scores)
    if best_mask is None:
        return False

    best_mask = clean_mask(best_mask)

    masked = apply_mask_white_bg(image_bgr, best_mask)
    cropped = crop_from_mask(masked, best_mask, padding=CROP_PADDING)

    if cropped is None:
        return False

    out_path = get_output_path(image_path, INPUT_DIR, OUTPUT_DIR)
    ensure_dir(out_path.parent)

    cv2.imwrite(str(out_path), cropped)
    return True


# =========================
# MAIN
# =========================
def main():
    ensure_dir(OUTPUT_DIR)

    predictor = load_model()

    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTS:
                image_paths.append(Path(root) / f)

    print(f"[INFO] Found {len(image_paths)} images")

    success, failed = 0, 0

    for i, img_path in enumerate(image_paths, 1):
        try:
            if process_image(img_path, predictor):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            failed += 1

        if i % 100 == 0:
            print(f"[PROGRESS] {i}/{len(image_paths)} | Success: {success} | Failed: {failed}")

    print(f"\n[DONE] Success: {success} | Failed: {failed}")


if __name__ == "__main__":
    main()