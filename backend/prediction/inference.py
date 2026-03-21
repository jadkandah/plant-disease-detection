"""
Real AI inference using the trained ResNet-50 plant disease model.

The model is loaded once (singleton) and reused for all predictions.
Input: PIL Image  →  Output: (class_key: str, confidence: float)
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from io import BytesIO

# ──────────────────────────────────────────────
# Model configuration (must match training script exactly)
# ──────────────────────────────────────────────
IMAGE_SIZE = 384
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Path to the trained .pth weights
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "dissdetector", "resnet_50_plant_disease.pth"
)

# ──────────────────────────────────────────────
# Class mapping — 45 classes, sorted alphabetically.
# This is the exact output of build_shared_mapping()
# from the training script scanning jordan_dataset folders.
# The model was trained WITHOUT Eggplant classes.
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
    "Orange___Citrus_greening",
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

NUM_CLASSES = len(CLASS_NAMES)  # 45

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
# Singleton model loader
# ──────────────────────────────────────────────
_model = None

def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def _load_model():
    """Load the model once and cache it globally."""
    global _model
    if _model is not None:
        return _model

    device = _get_device()
    print(f"[inference] Loading ResNet-50 model on {device}...")
    print(f"[inference] Model path: {MODEL_PATH}")
    print(f"[inference] Number of classes: {NUM_CLASSES}")

    # Build the same architecture as training
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # Load trained weights
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model = model
    print("[inference] Model loaded successfully!")
    return _model


def predict_image(image_file) -> tuple[str, float]:
    """
    Run inference on a Django UploadedFile or file-like object.

    Returns:
        (class_key, confidence) — the predicted class name and softmax probability.
    """
    model = _load_model()
    device = _get_device()

    # Read the uploaded file into a PIL Image
    image_bytes = image_file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Apply the same transforms used during validation
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    class_key = CLASS_NAMES[predicted_idx.item()]
    conf = round(confidence.item(), 4)

    print(f"[inference] Predicted: {class_key} (confidence: {conf})")
    return class_key, conf
