import os
import cv2
import numpy as np
from pathlib import Path

from mobile_sam import sam_model_registry, SamPredictor

# =========================
# CONFIG
# =========================
ROOT = Path("/Users/sanadmadani/Desktop/plant-disease-detection")


INPUT_DIR = ROOT / "jordan_dataset2"/"test"     # folder of input images
OUTPUT_DIR = ROOT / "green_seg_sam_output"    # folder where results will be saved

CHECKPOINT_PATH = ROOT / "mobile_sam.pt"

MODEL_TYPE = "vit_t"   # for MobileSAM
DEVICE = "cpu"         # change to "cuda" or "mps" if available

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_PADDING = 15


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def get_center_prompt_box(image):
    """
    Create a central prompt box automatically.
    Since your leaf is usually near center-ish and on plain background,
    this is a decent zero-manual-prompt start.
    """
    h, w = image.shape[:2]

    x1 = int(w * 0.10)
    y1 = int(h * 0.15)
    x2 = int(w * 0.90)
    y2 = int(h * 0.85)

    return np.array([x1, y1, x2, y2])


def choose_best_mask(masks, scores):
    """
    Pick best mask by score, but also prefer masks that are not absurdly huge.
    """
    best_idx = None
    best_value = -1

    for i, (mask, score) in enumerate(zip(masks, scores)):
        area = mask.sum()
        if area == 0:
            continue

        # Penalize overly huge masks
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

    # Keep largest contour only
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

    cropped = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    return cropped, cropped_mask, (x1, y1, x2, y2)


def apply_mask_white_bg(image, mask):
    background = np.full_like(image, 255)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = np.where(mask_3ch == 255, image, background)
    return out


def process_image(image_path, predictor, out_dir):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    box = get_center_prompt_box(image_bgr)

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    best_mask = choose_best_mask(masks, scores)
    if best_mask is None:
        print(f"[WARNING] No SAM mask found: {image_path}")
        return False

    best_mask = clean_mask(best_mask)

    masked = apply_mask_white_bg(image_bgr, best_mask)
    crop_data = crop_from_mask(masked, best_mask, padding=CROP_PADDING)

    stem = Path(image_path).stem

    mask_dir = os.path.join(out_dir, "masks")
    crop_dir = os.path.join(out_dir, "cropped")
    overlay_dir = os.path.join(out_dir, "overlay")

    ensure_dir(mask_dir)
    ensure_dir(crop_dir)
    ensure_dir(overlay_dir)

    cv2.imwrite(os.path.join(mask_dir, f"{stem}_mask.png"), best_mask)

    if crop_data is None:
        print(f"[WARNING] Could not crop: {image_path}")
        return False

    cropped_img, cropped_mask, bbox = crop_data
    x1, y1, x2, y2 = bbox

    debug = image_bgr.copy()
    cv2.rectangle(debug, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(crop_dir, f"{stem}_crop.png"), cropped_img)
    cv2.imwrite(os.path.join(overlay_dir, f"{stem}_debug.png"), debug)

    return True


def main():
    ensure_dir(OUTPUT_DIR)

    predictor = load_model()

    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTS:
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images")

    success = 0
    for img_path in image_paths:
        try:
            ok = process_image(img_path, predictor, OUTPUT_DIR)
            if ok:
                success += 1
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")

    print(f"Done. Success: {success}/{len(image_paths)}")


if __name__ == "__main__":
    main()