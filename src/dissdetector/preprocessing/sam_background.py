from pathlib import Path

import cv2
import numpy as np
import torch

from mobile_sam import SamPredictor, sam_model_registry


DEFAULT_MODEL_TYPE = "vit_t"
DEFAULT_CROP_PADDING = 15


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sam_predictor(
    checkpoint_path: Path,
    model_type: str = DEFAULT_MODEL_TYPE,
    device: str | None = None,
) -> SamPredictor:
    resolved_device = device or get_default_device()

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=resolved_device)
    sam.eval()
    return SamPredictor(sam)


def get_center_prompt_box(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    return np.array(
        [
            int(width * 0.10),
            int(height * 0.15),
            int(width * 0.90),
            int(height * 0.85),
        ]
    )


def choose_best_mask(masks, scores):
    best_idx = None
    best_value = -1.0

    for index, (mask, score) in enumerate(zip(masks, scores)):
        area = mask.sum()
        if area == 0:
            continue

        quality = float(score) - 0.000001 * area
        if quality > best_value:
            best_value = quality
            best_idx = index

    if best_idx is None:
        return None

    return masks[best_idx].astype(np.uint8) * 255


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel_small = np.ones((5, 5), np.uint8)
    kernel_big = np.ones((7, 7), np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_big)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)
    output = np.zeros_like(mask)
    cv2.drawContours(output, [largest], -1, 255, thickness=-1)
    return output


def crop_from_mask(image: np.ndarray, mask: np.ndarray, padding: int = DEFAULT_CROP_PADDING):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, width, height = cv2.boundingRect(coords)

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + width + padding, image.shape[1])
    y2 = min(y + height + padding, image.shape[0])

    return image[y1:y2, x1:x2]


def apply_mask_white_bg(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    background = np.full_like(image, 255)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(mask_3ch == 255, image, background)


def remove_background(
    image_bgr: np.ndarray,
    predictor: SamPredictor,
    crop_padding: int = DEFAULT_CROP_PADDING,
):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(
        box=get_center_prompt_box(image_bgr),
        multimask_output=True,
    )

    best_mask = choose_best_mask(masks, scores)
    if best_mask is None:
        return None

    best_mask = clean_mask(best_mask)
    masked = apply_mask_white_bg(image_bgr, best_mask)
    return crop_from_mask(masked, best_mask, padding=crop_padding)
