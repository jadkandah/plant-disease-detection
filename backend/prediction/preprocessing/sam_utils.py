import os
import numpy as np
_SAM_AVAILABLE = False
_predictor = None

try:
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    print("[sam_utils] MobileSAM not installed — leaf extraction disabled. "
          "Install with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")


SAM_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "mobile_sam.pt"
)


def is_sam_available():
    return _SAM_AVAILABLE and os.path.exists(SAM_CHECKPOINT)


def load_sam():
    global _predictor
    if _predictor is not None:
        return _predictor

    if not _SAM_AVAILABLE:
        print("[sam_utils] MobileSAM not available.")
        return None

    if not os.path.exists(SAM_CHECKPOINT):
        print(f"[sam_utils] SAM checkpoint not found at: {SAM_CHECKPOINT}")
        print("[sam_utils] Download with: wget -O mobile_sam.pt "
              "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt")
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = sam_model_registry["vit_t"](checkpoint=SAM_CHECKPOINT)
        model.to(device)
        _predictor = SamPredictor(model)
        print(f"[sam_utils] MobileSAM loaded successfully on {device}")
        return _predictor
    except Exception as e:
        print(f"[sam_utils] Failed to load SAM: {e}")
        return None


def extract_leaf(image: np.ndarray) -> np.ndarray:
    predictor = load_sam()
    if predictor is None:
        print("[sam_utils] SAM not available — returning original image")
        return image

    try:

        import cv2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        h, w, _ = image.shape

        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask = masks[0]


        white_bg = np.ones_like(image) * 255
        result = np.where(mask[:, :, None], image, white_bg).astype(np.uint8)

        print(f"[sam_utils] Leaf extracted successfully (mask coverage: {mask.mean():.1%})")
        return result

    except Exception as e:
        print(f"[sam_utils] Leaf extraction failed: {e} — returning original image")
        return image
