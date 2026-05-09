import cv2
import numpy as np


def _build_leaf_color_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green, yellow, and brown cover healthy and stressed crop leaves.
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])

    yellow_lower = np.array([20, 40, 40])
    yellow_upper = np.array([35, 255, 255])

    brown_lower = np.array([10, 50, 20])
    brown_upper = np.array([20, 255, 200])

    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)

    leaf_mask = cv2.bitwise_or(green_mask, yellow_mask)
    leaf_mask = cv2.bitwise_or(leaf_mask, brown_mask)

    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    return hsv, green_mask, yellow_mask, brown_mask, leaf_mask


def _largest_component_stats(mask, total_pixels):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            'largest_ratio': 0,
            'largest_leaf_fraction': 0,
            'extent': 0,
            'solidity': 0,
        }

    largest = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest)
    leaf_area = np.count_nonzero(mask)

    x, y, w, h = cv2.boundingRect(largest)
    bbox_area = max(w * h, 1)
    hull = cv2.convexHull(largest)
    hull_area = max(cv2.contourArea(hull), 1)

    return {
        'largest_ratio': largest_area / total_pixels,
        'largest_leaf_fraction': largest_area / max(leaf_area, 1),
        'extent': largest_area / bbox_area,
        'solidity': largest_area / hull_area,
    }


def _leaf_texture_stats(image, hsv, mask):
    leaf_pixels = mask > 0
    leaf_count = np.count_nonzero(leaf_pixels)
    if leaf_count == 0:
        return {'edge_ratio': 0, 'saturation_std': 0, 'value_std': 0}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((7, 7), np.uint8)
    interior_pixels = cv2.erode(mask, kernel) > 0
    interior_count = np.count_nonzero(interior_pixels)
    texture_pixels = interior_pixels if interior_count else leaf_pixels
    texture_count = interior_count if interior_count else leaf_count
    edge_ratio = np.count_nonzero((edges > 0) & texture_pixels) / texture_count

    saturation_std = float(np.std(hsv[:, :, 1][leaf_pixels]))
    value_std = float(np.std(hsv[:, :, 2][leaf_pixels]))

    return {
        'edge_ratio': edge_ratio,
        'saturation_std': saturation_std,
        'value_std': value_std,
    }


def is_leaf_image(image):
    hsv, green_mask, yellow_mask, brown_mask, leaf_mask = _build_leaf_color_mask(image)

    total_pixels = image.shape[0] * image.shape[1]
    green_ratio = np.sum(green_mask > 0) / total_pixels
    yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
    brown_ratio = np.sum(brown_mask > 0) / total_pixels
    leaf_ratio = np.sum(leaf_mask > 0) / total_pixels
    component_stats = _largest_component_stats(leaf_mask, total_pixels)
    texture_stats = _leaf_texture_stats(image, hsv, leaf_mask)

    print(
        f"[LeafCheck] green={green_ratio:.2f}, yellow={yellow_ratio:.2f}, "
        f"brown={brown_ratio:.2f}, total={leaf_ratio:.2f}, "
        f"largest={component_stats['largest_ratio']:.2f}, "
        f"edge={texture_stats['edge_ratio']:.3f}, "
        f"s_std={texture_stats['saturation_std']:.1f}, "
        f"v_std={texture_stats['value_std']:.1f}"
    )

    has_green_leaf_area = green_ratio > 0.04
    has_stressed_leaf_area = (
        green_ratio > 0.015
        and (yellow_ratio + brown_ratio > 0.06)
    )
    has_enough_leaf_pixels = leaf_ratio > 0.08
    has_contiguous_leaf_region = (
        component_stats['largest_ratio'] > 0.035
        and component_stats['largest_leaf_fraction'] > 0.35
    )
    has_natural_variation = (
        texture_stats['edge_ratio'] > 0.006
        or texture_stats['saturation_std'] > 18
        or texture_stats['value_std'] > 18
    )
    is_flat_surface = (
        leaf_ratio > 0.55
        and component_stats['largest_leaf_fraction'] > 0.75
        and not has_natural_variation
    )
    is_uniform_artificial_shape = (
        component_stats['extent'] > 0.90
        and component_stats['solidity'] > 0.96
        and not has_natural_variation
    )
    is_leaf = (
        (has_green_leaf_area or has_stressed_leaf_area)
        and has_enough_leaf_pixels
        and has_contiguous_leaf_region
        and not is_flat_surface
        and not is_uniform_artificial_shape
    )

    print(
        f"[LeafCheck] accepted={is_leaf}, green_leaf={has_green_leaf_area}, "
        f"stressed_leaf={has_stressed_leaf_area}, "
        f"contiguous={has_contiguous_leaf_region}, "
        f"natural_variation={has_natural_variation}"
    )

    return is_leaf, leaf_ratio


def is_leaf_color(image, threshold=0.3):
    return is_leaf_image(image)
