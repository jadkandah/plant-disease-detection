import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image




MODEL_PATH = "saved_models/mobilenet_v3_small_512_epochs25_full_data_set.pth"
TRAIN_DIR = "jordan_dataset/train"
TEST_DIR = "jordan_dataset/test"

IMAGE_SIZE = 512
IMAGES_PER_CLASS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("src/dissdetector/GRAD_CAM/GRAD_CAM_Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}




def find_leaf_class_dirs(root_dir):
    root = Path(root_dir)
    class_dirs = []
    for current_dir, _, files in os.walk(root):
        current_path = Path(current_dir)
        has_image = any((current_path / f).suffix.lower() in VALID_EXTS for f in files)
        if has_image:
            class_dirs.append(current_path.relative_to(root).as_posix())
    return sorted(class_dirs)

CLASS_NAMES = find_leaf_class_dirs(TRAIN_DIR)




model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))

loaded = torch.load(MODEL_PATH, map_location=DEVICE)

try:
    model.load_state_dict(loaded)
except Exception:
    if isinstance(loaded, dict) and "state_dict" in loaded:
        model.load_state_dict(loaded["state_dict"])
    elif isinstance(loaded, dict) and "model_state_dict" in loaded:
        model.load_state_dict(loaded["model_state_dict"])
    else:
        model = loaded

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




def get_images_by_class(root_dir):
    root = Path(root_dir)
    class_to_images = {}

    for cls in CLASS_NAMES:
        cls_path = root / cls
        if not cls_path.exists():
            continue

        images = [p for p in cls_path.iterdir() if p.suffix.lower() in VALID_EXTS]

        if len(images) > 0:
            class_to_images[cls] = images

    return class_to_images

class_images = get_images_by_class(TEST_DIR)




target_layers = [model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)




for cls, images in class_images.items():
    print(f"\nProcessing class: {cls}")


    sample_images = random.sample(images, min(IMAGES_PER_CLASS, len(images)))

    for img_path in sample_images:
        pil_img = Image.open(img_path).convert("RGB")
        pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))

        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        rgb_img = np.array(pil_img).astype(np.float32) / 255.0


        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASS_NAMES[pred_idx]
            pred_conf = probs[0, pred_idx].item()


        grayscale_cam = cam(input_tensor=input_tensor)[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


        safe_cls = cls.replace("/", "_")
        save_name = f"{safe_cls}__{img_path.stem}__pred_{pred_label.replace('/', '_')}.jpg"
        save_path = OUTPUT_DIR / save_name

        cv2.imwrite(str(save_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        print(f"Saved: {save_path.name} | pred={pred_label} ({pred_conf:.3f})")
