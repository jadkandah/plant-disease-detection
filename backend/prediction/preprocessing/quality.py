"""
Image quality checks for plant disease detection.

Rejects images that are blurry, too dark, too bright, low contrast,
corrupted, or completely black — prevents garbage-in/garbage-out.
"""
import cv2
import numpy as np


def is_blurry(image, threshold=100):
    """Check if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_too_dark(image, threshold=50):
    """Check if image is too dark (mean brightness below threshold)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold


def is_too_bright(image, threshold=200):
    """Check if image is too bright (mean brightness above threshold)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > threshold


def is_low_contrast(image, threshold=20):
    """Check if image has too little contrast (low std deviation)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std() < threshold


def is_black(image, threshold=10):
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
