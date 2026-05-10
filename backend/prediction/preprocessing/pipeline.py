"""
Backend preprocessing pipeline.

Common image quality checks (blur, brightness, contrast) are now performed
on the **frontend** before the image is uploaded.  This pipeline handles
only backend-specific preprocessing:

  1. Decode the uploaded bytes into an OpenCV image
  2. SAM background removal (online mode only)
  3. (Future: any additional backend-only transforms)

The quality module is kept in the codebase as a safety-net fallback and
can be re-enabled by uncommenting the guard below.
"""

import cv2
import numpy as np
from .sam_utils import extract_leaf


def preprocess_image(file, mode="offline"):
    """
    Preprocess an uploaded image file for model inference.

    Args:
        file: A Django UploadedFile (or anything with a .read() method).
        mode: "online" | "offline"

    Returns:
        (image, status_string)
        image is a BGR numpy array (or None on failure).
    """
    # ── Decode ──────────────────────────────────────────────────
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return None, "Rejected: corrupted — could not decode image"

    print(f"[pipeline] Image decoded: {image.shape}, mode={mode}")

    # ── Quality check (SKIPPED — handled by the frontend) ──────
    # If you ever need a backend fallback, uncomment these lines:
    # from .quality import check_quality
    # is_valid, reason = check_quality(image)
    # if not is_valid:
    #     print(f"[pipeline] Quality check FAILED: {reason}")
    #     return None, f"Rejected: {reason}"

    # ── Online mode → SAM leaf extraction (backend-only) ───────
    if mode == "online":
        print("[pipeline] Running SAM background removal (backend-only)")
        image = extract_leaf(image)
    else:
        print("[pipeline] Skipping SAM preprocessing (offline mode)")

    # NOTE: We do NOT resize here. The inference transforms handle
    # resizing to the model's expected input size (512×512).

    return image, "OK"
