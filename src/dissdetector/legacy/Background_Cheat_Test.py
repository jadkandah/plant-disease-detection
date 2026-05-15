import os
from pathlib import Path
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms, models


MODEL_PATH = "saved_models/mobilenet_v3_small_512_epochs25_full_data_set.pth"
DATASET_ROOT = "jordan_dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

IMAGE_SIZE = 512
BLUR_RADIUS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTS

def find_leaf_class_dirs(root_dir):







    root = Path(root_dir)
    class_dirs = []

    for current_dir, _, files in os.walk(root):
        current_path = Path(current_dir)
        has_image = any((current_path / f).suffix.lower() in VALID_EXTS for f in files)
        if has_image:
            rel_path = current_path.relative_to(root).as_posix()
            class_dirs.append(rel_path)

    class_dirs = sorted(class_dirs)
    return class_dirs

def get_all_test_images(root_dir):



    root = Path(root_dir)
    image_paths = []
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            image_paths.append(p)
    return sorted(image_paths)

def get_true_label_from_path(img_path, split_root):





    rel = img_path.relative_to(split_root)
    return rel.parent.as_posix()

def make_blurry_image(image: Image.Image, blur_radius: float = 8.0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


CLASS_NAMES = find_leaf_class_dirs(TRAIN_DIR)

if len(CLASS_NAMES) == 0:
    raise ValueError(f"No class folders with images found under: {TRAIN_DIR}")

print(f"Found {len(CLASS_NAMES)} classes:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"{i}: {cls}")


model = models.mobilenet_v3_small(weights=None)


in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))


loaded = torch.load(MODEL_PATH, map_location=DEVICE)

try:
    model.load_state_dict(loaded)
    print("Loaded model as plain state_dict.")
except Exception:
    try:
        if isinstance(loaded, dict) and "state_dict" in loaded:
            model.load_state_dict(loaded["state_dict"])
            print("Loaded model from checkpoint['state_dict'].")
        elif isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
            print("Loaded model from checkpoint['model_state_dict'].")
        else:

            model = loaded
            print("Loaded full saved model object.")
    except Exception as e:
        raise RuntimeError(f"Could not load model properly: {e}")

model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, CLASS_NAMES[pred_idx], confidence, probs[0].cpu().tolist()


test_images = get_all_test_images(TEST_DIR)

if len(test_images) == 0:
    raise ValueError(f"No test images found under: {TEST_DIR}")

correct_normal = 0
correct_blurry = 0
total = 0

for img_path in test_images:
    true_label = get_true_label_from_path(img_path, Path(TEST_DIR))

    img = Image.open(img_path).convert("RGB")
    blurry_img = make_blurry_image(img, blur_radius=BLUR_RADIUS)

    _, pred_normal, conf_normal, _ = predict_image(img)
    _, pred_blurry, conf_blurry, _ = predict_image(blurry_img)

    normal_ok = pred_normal == true_label
    blurry_ok = pred_blurry == true_label

    correct_normal += int(normal_ok)
    correct_blurry += int(blurry_ok)
    total += 1

    print("=" * 80)
    print(f"Image:       {img_path.name}")
    print(f"True label:  {true_label}")
    print(f"Normal pred: {pred_normal} | confidence={conf_normal:.4f} | correct={normal_ok}")
    print(f"Blurry pred: {pred_blurry} | confidence={conf_blurry:.4f} | correct={blurry_ok}")

print("\n" + "#" * 80)
print(f"Total images tested: {total}")
print(f"Normal accuracy: {correct_normal}/{total} = {correct_normal / total:.4%}")
print(f"Blurry accuracy: {correct_blurry}/{total} = {correct_blurry / total:.4%}")
print("#" * 80)
