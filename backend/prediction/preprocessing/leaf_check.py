import cv2
import numpy as np


def is_leaf_color(image, threshold=0.3):
    """
    Detect if image contains leaf-like colors:
    green, yellow, brown.

    The threshold argument is kept for API compatibility; detection now uses
    per-color minimums plus a total leaf-color ratio to avoid rejecting mixed
    leaf colors against natural backgrounds.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 🟢 Green range
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])

    # 🟡 Yellow range
    yellow_lower = np.array([20, 40, 40])
    yellow_upper = np.array([35, 255, 255])

    # 🟤 Brown range
    brown_lower = np.array([10, 50, 20])
    brown_upper = np.array([20, 255, 200])

    # Create masks
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

    # Combine all
    combined_mask = cv2.bitwise_or(green_mask, yellow_mask)
    combined_mask = cv2.bitwise_or(combined_mask, brown_mask)

    total_pixels = image.shape[0] * image.shape[1]
    green_ratio = np.sum(green_mask > 0) / total_pixels
    yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
    brown_ratio = np.sum(brown_mask > 0) / total_pixels
    leaf_ratio = np.sum(combined_mask > 0) / total_pixels

    print(
        f"[LeafCheck] green={green_ratio:.2f}, yellow={yellow_ratio:.2f}, "
        f"brown={brown_ratio:.2f}, total={leaf_ratio:.2f}"
    )

    has_green_leaf_area = green_ratio > 0.05
    has_stressed_leaf_area = (
        green_ratio > 0.03
        and (yellow_ratio > 0.05 or brown_ratio > 0.05)
    )
    has_enough_leaf_pixels = leaf_ratio > 0.08
    is_leaf = (has_green_leaf_area or has_stressed_leaf_area) and has_enough_leaf_pixels

    print(
        f"[LeafCheck] accepted={is_leaf}, green_leaf={has_green_leaf_area}, "
        f"stressed_leaf={has_stressed_leaf_area}"
    )

    return is_leaf, leaf_ratio
