import cv2
import numpy as np
from .quality import check_quality
from .sam_utils import extract_leaf
from .leaf_check import is_leaf_image


def preprocess_image(file, mode="offline"):
    # Read file bytes into cv2 image
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return None, "Rejected: corrupted — could not decode image"

    print(f"[pipeline] Image decoded: {image.shape}, mode={mode}")

    # 1. Quality check
    print("[pipeline] Running quality check")
    is_valid, reason = check_quality(image)
    if not is_valid:
        print(f"[pipeline] Quality check FAILED: {reason}")
        return None, f"Rejected: {reason}"

    # Reject obvious non-leaf inputs before classification.
    is_leaf, ratio = is_leaf_image(image)
    
    if not is_leaf:
        return None, f"Rejected: Not a leaf (ratio={ratio:.2f})"

    # 2. Online mode → SAM leaf extraction (if available)
    if mode == "online":
        print("[pipeline] Running SAM preprocessing")
        image = extract_leaf(image)
    else:
        print("[pipeline] Skipping SAM preprocessing (offline mode)")

    # NOTE: We do NOT resize here. The inference transforms handle
    # resizing to the model's expected input size (512x512).

    return image, "OK"
