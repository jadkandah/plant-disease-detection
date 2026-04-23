"""
Image quality checks for plant disease detection.

Rejects only truly unusable images: completely black, fully white,
corrupted, or extremely blurry. Normal phone photos always pass.
"""
import cv2
import numpy as np


def is_blurry(image, threshold=15):
    """Only reject extremely blurry images (threshold 15 is very permissive)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_too_dark(image, threshold=15):
    """Only reject nearly black images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


def is_too_bright(image, threshold=245):
    """Only reject nearly white/overexposed images."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > threshold


def is_low_contrast(image, threshold=8):
    """Only reject images with almost zero contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std() < threshold


def is_black(image, threshold=5):
    """Check if image is essentially all black."""
    return image.mean() < threshold


def check_quality(image):
    """
    Run all quality checks on a BGR image (numpy array).

    Returns:
        (is_valid: bool, reason: str)
        - True, "good" if the image passes all checks
        - False, "<reason>" if the image fails
    """
    if image is None:
        return False, "corrupted"
    if image.size == 0:
        return False, "empty"
    if is_black(image):
        return False, "black"
    if is_blurry(image):
        return False, "blurry"
    if is_too_dark(image):
        return False, "too_dark"
    if is_too_bright(image):
        return False, "too_bright"
    if is_low_contrast(image):
        return False, "low_contrast"
    return True, "good"
