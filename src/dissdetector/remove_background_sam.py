import os
import cv2
import numpy as np
from pathlib import Path

from mobile_sam import sam_model_registry, SamPredictor

# =========================
# CONFIG
# =========================
# ROOT = Path("/Users/sanadmadani/Desktop/plant-disease-detection")
ROOT = Path("/home/jad/plant-disease-detection")

INPUT_DIR = ROOT / "jordan_dataset" / "train"
OUTPUT_DIR = ROOT / "jordan_dataset" / "train_images_background_removed"

CHECKPOINT_PATH = ROOT / "mobile_sam.pt"

MODEL_TYPE = "vit_t"   # for MobileSAM
DEVICE = "cpu"         # change to "cuda" or "mps" if available

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_PADDING = 15

# =========================
# CLASS NAME MAP
# Converts original class-folder names into:
#   Plant / Disease
# =========================
CLASS_NAME_MAP = {
    "Healthy Wheat": ("Wheat", "healthy"),
    "Wheat aphid": ("Wheat", "Aphid"),
    "Wheat black rust": ("Wheat", "Black_rust"),
    "Wheat Brown leaf Rust": ("Wheat", "Brown_leaf_Rust"),
    "Wheat leaf blight": ("Wheat", "Leaf_blight"),
    "Wheat mite": ("Wheat", "Mite"),
    "Wheat powdery mildew": ("Wheat", "Powdery_mildew"),
    "Wheat scab": ("Wheat", "Scab"),
    "Wheat Stem fly": ("Wheat", "Stem_fly"),
    "Wheat___Yellow_Rust": ("Wheat", "Yellow_Rust"),

    "Cauliflower_Bacterial_spot_rot": ("Cauliflower", "Bacterial_spot_rot"),
    "Cauliflower_Black_Rot": ("Cauliflower", "Black_Rot"),
    "Cauliflower_Downy_Mildew": ("Cauliflower", "Downy_Mildew"),
    "Cauliflower_Healthy": ("Cauliflower", "healthy"),

    "EggPlant_Healthy_Leaf": ("Eggplant", "healthy"),
    "EggPlant_Insect_Pest_Disease": ("Eggplant", "Insect_Pest_Disease"),
    "EggPlant_Leaf_Spot_Disease": ("Eggplant", "Leaf_Spot_Disease"),
    "EggPlant_Mosaic_Virus_Disease": ("Eggplant", "Mosaic_Virus_Disease"),
    "EggPlant_Small_Leaf_Disease": ("Eggplant", "Small_Leaf_Disease"),
    "EggPlant_White_Mold_Disease": ("Eggplant", "White_Mold_Disease"),
    "EggPlant_Wilt_Disease": ("Eggplant", "Wilt_Disease"),

    "Apple___Apple_scab": ("Apple", "Apple_scab"),
    "Apple___Black_rot": ("Apple", "Black_rot"),
    "Apple___Cedar_apple_rust": ("Apple", "Cedar_apple_rust"),
    "Apple___healthy": ("Apple", "healthy"),

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Maize", "Cercospora_leaf_spot_Gray_leaf_spot"),
    "Corn_(maize)___Common_rust_": ("Maize", "Common_rust"),
    "Corn_(maize)___healthy": ("Maize", "healthy"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Maize", "Northern_Leaf_Blight"),

    "Grape___Black_rot": ("Grape", "Black_rot"),
    "Grape___Esca_(Black_Measles)": ("Grape", "Esca_Black_Measles"),
    "Grape___healthy": ("Grape", "healthy"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Grape", "Leaf_blight_Isariopsis_Leaf_Spot"),

    "Orange___healthy": ("Orange", "healthy"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Orange", "Citrus_greening"),
    "Orange___Canker": ("Orange", "Canker"),
    "Orange___Black_spot": ("Orange", "Black_spot"),

    "Peach___Bacterial_spot": ("Peach", "Bacterial_spot"),
    "Peach___healthy": ("Peach", "healthy"),

    "Potato___Early_blight": ("Potato", "Early_blight"),
    "Potato___healthy": ("Potato", "healthy"),
    "Potato___Late_blight": ("Potato", "Late_blight"),

    "Tomato___Bacterial_spot": ("Tomato", "Bacterial_spot"),
    "Tomato___Early_blight": ("Tomato", "Early_blight"),
    "Tomato___healthy": ("Tomato", "healthy"),
    "Tomato___Late_blight": ("Tomato", "Late_blight"),
    "Tomato___Leaf_Mold": ("Tomato", "Leaf_Mold"),
    "Tomato___Septoria_leaf_spot": ("Tomato", "Septoria_leaf_spot"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomato", "Spider_mites"),
    "Tomato___Target_Spot": ("Tomato", "Target_Spot"),
    "Tomato___Tomato_mosaic_virus": ("Tomato", "Mosaic_virus"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomato", "Yellow_Leaf_Curl_Virus"),

    "aculus_olearius": ("Olive", "Aculus_olearius_mite"),
    "Healthy": ("Olive", "healthy"),
    "olive_peacock_spot": ("Olive", "Peacock_spot"),
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model():
    sam = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT_PATH))
    sam.to(device=DEVICE)
    sam.eval()
    predictor = SamPredictor(sam)
    return predictor


def get_center_prompt_box(image):
    """
    Create a central prompt box automatically.
    Since the leaf is usually near the center and on a plain background,
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

    cropped = image[y1:y2, x1:x2]
    return cropped


def apply_mask_white_bg(image, mask):
    background = np.full_like(image, 255)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    out = np.where(mask_3ch == 255, image, background)
    return out


def parse_class_folder_name(class_name: str):
    """
    Convert a class folder name into:
        plant_name, disease_name

    Priority:
    1) exact lookup in CLASS_NAME_MAP
    2) split on '___'
    3) fallback to class_name / unknown
    """
    if class_name in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[class_name]

    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
        return plant.strip(), disease.strip()

    return class_name.strip(), "unknown"


def get_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    """
    Input structure:
        train/ClassFolder/image.jpg

    Output structure:
        train_images_background_removed/PlantName/DiseaseName/image.png
    """
    rel_path = input_path.relative_to(input_root)
    parts = rel_path.parts

    if len(parts) < 2:
        # fallback
        return output_root / "unknown" / "unknown" / input_path.with_suffix(".png").name

    class_folder = parts[0]
    filename = Path(parts[-1]).with_suffix(".png").name

    plant_name, disease_name = parse_class_folder_name(class_folder)

    return output_root / plant_name / disease_name / filename


def process_image(image_path, predictor, input_root, output_root):
    image_path = Path(image_path)

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
    cropped_img = crop_from_mask(masked, best_mask, padding=CROP_PADDING)

    if cropped_img is None:
        print(f"[WARNING] Could not crop: {image_path}")
        return False

    out_path = get_output_path(image_path, input_root, output_root)
    ensure_dir(out_path.parent)

    cv2.imwrite(str(out_path), cropped_img)
    return True


def main():
    ensure_dir(OUTPUT_DIR)

    predictor = load_model()

    image_paths = []
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if Path(f).suffix.lower() in VALID_EXTS:
                image_paths.append(Path(root) / f)

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images")

    success = 0
    failed = 0

    for i, img_path in enumerate(image_paths, start=1):
        try:
            ok = process_image(img_path, predictor, INPUT_DIR, OUTPUT_DIR)
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {img_path}: {e}")

        if i % 100 == 0:
            print(f"Processed {i}/{len(image_paths)} | Success: {success} | Failed: {failed}")

    print(f"Done. Success: {success}/{len(image_paths)} | Failed: {failed}")


if __name__ == "__main__":
    main()