"""
Dual-model AI inference for plant disease detection.

- Offline model: MobileNetV3-Small (fast, lightweight, image-only)
- Online model:  ResNet50 image-only 512px (higher mIoU/lower val loss choice)

Both models are loaded once (singleton) and reused for all predictions.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# ──────────────────────────────────────────────
# Model configuration (must match training script exactly)
# ──────────────────────────────────────────────
IMAGE_SIZE = 512
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model weight paths
OFFLINE_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "saved_models",
    "mobilenet_v3_small_512_background_removed_epochs25.pth"
)
ONLINE_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "saved_models",
    "image_only_resnet50_background_removed_512_epochs40.pth"
)

# ──────────────────────────────────────────────
# 55 classes — sorted alphabetically, matching
# build_shared_mapping() from the jordan_dataset.
# Includes Eggplant + extended Orange classes.
# ──────────────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cauliflower___Bacterial_spot_rot",
    "Cauliflower___Black_Rot",
    "Cauliflower___Downy_Mildew",
    "Cauliflower___healthy",
    "Eggplant___Insect_Pest_Disease",
    "Eggplant___Leaf_Spot_Disease",
    "Eggplant___Mosaic_Virus_Disease",
    "Eggplant___Small_Leaf_Disease",
    "Eggplant___White_Mold_Disease",
    "Eggplant___Wilt_Disease",
    "Eggplant___healthy",
    "Grape___Black_rot",
    "Grape___Esca_Black_Measles",
    "Grape___Leaf_blight_Isariopsis_Leaf_Spot",
    "Grape___healthy",
    "Maize___Cercospora_leaf_spot_Gray_leaf_spot",
    "Maize___Common_rust",
    "Maize___Northern_Leaf_Blight",
    "Maize___healthy",
    "Olive___Aculus_olearius_mite",
    "Olive___Peacock_spot",
    "Olive___healthy",
    "Orange___Black_spot",
    "Orange___Canker",
    "Orange___Citrus_greening",
    "Orange___healthy",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Mosaic_virus",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
    "Wheat___Aphid",
    "Wheat___Black_rust",
    "Wheat___Brown_leaf_Rust",
    "Wheat___Leaf_blight",
    "Wheat___Mite",
    "Wheat___Powdery_mildew",
    "Wheat___Scab",
    "Wheat___Stem_fly",
    "Wheat___Yellow_Rust",
    "Wheat___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)  # 55

# ──────────────────────────────────────────────
# Inference transform (matches val_test_transforms from training)
# ──────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


# ──────────────────────────────────────────────
# Singleton model loaders
# ──────────────────────────────────────────────
_offline_model = None
_online_model = None


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_offline_model():
    """Load MobileNetV3-Small model (offline mode — fast, lightweight)."""
    global _offline_model
    if _offline_model is not None:
        return _offline_model

    device = _get_device()
    print(f"[inference] Loading MobileNetV3-Small (offline) on {device}...")

    if not os.path.exists(OFFLINE_MODEL_PATH):
        raise FileNotFoundError(f"Offline model not found: {OFFLINE_MODEL_PATH}")

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    state_dict = torch.load(OFFLINE_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _offline_model = model
    print("[inference] MobileNetV3-Small (offline) loaded successfully!")
    return _offline_model


def _load_online_model():
    """Load ResNet50 image-only model (online mode — 512px background-removed model)."""
    global _online_model
    if _online_model is not None:
        return _online_model

    device = _get_device()
    print(f"[inference] Loading ResNet50 image-only 512 (online) on {device}...")

    if not os.path.exists(ONLINE_MODEL_PATH):
        raise FileNotFoundError(f"Online model not found: {ONLINE_MODEL_PATH}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(ONLINE_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _online_model = model
    print("[inference] ResNet50 image-only 512 (online) loaded successfully!")
    return _online_model


def predict_from_array(image_array: np.ndarray, mode: str = "offline") -> tuple[str, float]:
    """
    Run inference on a preprocessed numpy array (BGR format from cv2).

    Args:
        image_array: BGR numpy array from cv2/preprocessing pipeline
        mode: "online" (ResNet50 image-only 512) or "offline" (MobileNetV3-Small)

    Returns:
        (class_key, confidence)
    """
    device = _get_device()

    # Convert BGR → RGB → PIL
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = inference_transform(pil_image).unsqueeze(0).to(device)

    if mode == "online":
        # ── Online: image-only ResNet50 ──
        print("[inference] Using ResNet50 image-only 512")
        model = _load_online_model()

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        model_name = "ResNet50-ImageOnly-512"
    else:
        # ── Offline: image only ──
        print("[inference] Using MobileNetV3 Small")
        model = _load_offline_model()

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        model_name = "MobileNetV3-Small"

    class_key = CLASS_NAMES[predicted_idx.item()]
    conf = round(confidence.item(), 4)

    print(f"[inference] [{model_name}] Predicted: {class_key} (confidence: {conf})")
    return class_key, conf


def predict_image(image_file, mode: str = "offline") -> tuple[str, float]:
    """
    Run inference on a Django UploadedFile or file-like object.
    Backward-compatible wrapper around predict_from_array.
    """
    image_bytes = image_file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return predict_from_array(image_array, mode=mode)
