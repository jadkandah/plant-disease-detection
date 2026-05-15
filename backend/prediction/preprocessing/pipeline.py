import cv2
import numpy as np
from .sam_utils import extract_leaf


def preprocess_image(file, mode="offline"):
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return None, "Rejected: corrupted — could not decode image"

    print(f"[pipeline] Image decoded: {image.shape}, mode={mode}")
    if mode == "online":
        print("[pipeline] Running SAM background removal (backend-only)")
        image = extract_leaf(image)
    else:
        print("[pipeline] Skipping SAM preprocessing (offline mode)")

    return image, "OK"
