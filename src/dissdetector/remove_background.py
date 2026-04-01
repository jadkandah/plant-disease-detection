import os
import cv2
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================

#ROOT = Path("/Users/sanadmadani/Desktop/plant-disease-detection")
ROOT = Path("/home/jad/plant-disease-detection")
INPUT_DIR = ROOT / "jordan_dataset2" / "train"
OUTPUT_DIR = ROOT / "no_background_images"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MIN_LEAF_AREA = 500
CROP_PADDING = 15


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img


def get_green_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green_1 = np.array([25, 20, 20], dtype=np.uint8)
    upper_green_1 = np.array([100, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_green_1, upper_green_1)

    lower_green_2 = np.array([15, 10, 20], dtype=np.uint8)
    upper_green_2 = np.array([40, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_green_2, upper_green_2)

    return cv2.bitwise_or(mask1, mask2)


def clean_mask(mask):
    kernel_small = np.ones((5, 5), np.uint8)
    kernel_big = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
    mask = cv2.dilate(mask, kernel_small, iterations=1)

    return mask


def keep_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(mask), None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= MIN_LEAF_AREA:
            clean = np.zeros_like(mask)
            cv2.drawContours(clean, [cnt], -1, 255, thickness=-1)
            return clean, cnt

    return np.zeros_like(mask), None


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
    return cropped


def apply_mask(image, mask):
    """
    Keep leaf, set background to white.
    """
    background = np.full_like(image, 255)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = np.where(mask_3ch == 255, image, background)
    return out


def process_one_image(image_path, out_dir):
    image = read_image(image_path)

    raw_mask = get_green_mask(image)
    cleaned_mask = clean_mask(raw_mask)
    final_mask, cnt = keep_largest_contour(cleaned_mask)

    if cnt is None:
        print(f"[WARNING] No good leaf found: {image_path}")
        return False

    final_img = apply_mask(image, final_mask)
    cropped_final = crop_from_mask(final_img, final_mask, padding=CROP_PADDING)

    if cropped_final is None:
        print(f"[WARNING] Could not crop: {image_path}")
        return False

    stem = Path(image_path).stem
    ensure_dir(out_dir)

    cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), cropped_final)
    return True


def main():
    ensure_dir(OUTPUT_DIR)

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
            ok = process_one_image(img_path, OUTPUT_DIR)
            if ok:
                success += 1
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")

    print(f"Done. Success: {success}/{len(image_paths)}")


if __name__ == "__main__":
    main()