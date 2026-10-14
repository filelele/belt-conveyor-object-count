import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import math
from typing import Sequence, Union
import re
import sys
sys.path.append("../model/ml-mobileone")
from mobileone import mobileone

def create_gamma_lut(gamma: float = 2.0) -> np.ndarray:
    """Precomputed lookup table for dark suppression / gamma curve."""
    return np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)


def resize_and_pad_to_224(crop_img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resizes the larger dimension to 224.
    Forces the smaller dimension to an EVEN integer to allow symmetric integer padding,
    then pads with black borders to produce a (224, 224, 3) image.
    """
    h, w = crop_img.shape[:2]
    if h == 0 or w == 0:
        return None

    if h >= w:
        new_h = target_size
        scale = target_size / h
        new_w = int(round(w * scale))

        # Enforce even number
        if new_w % 2 != 0:
            new_w += 1 if (new_w + 1 <= target_size) else -1
        new_w = max(2, min(new_w, target_size))

        resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_total = target_size - new_w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = cv2.copyMakeBorder(
            resized, 
            top=0, bottom=0, left=pad_left, right=pad_right, 
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        new_w = target_size
        scale = target_size / w
        new_h = int(round(h * scale))

        # Enforce even number
        if new_h % 2 != 0:
            new_h += 1 if (new_h + 1 <= target_size) else -1
        new_h = max(2, min(new_h, target_size))

        resized = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_total = target_size - new_h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = cv2.copyMakeBorder(
            resized, 
            top=pad_top, bottom=pad_bottom, left=0, right=0, 
            borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return padded


from typing import Optional, Sequence, Tuple, Union

def extract_and_crop_object(
    img: np.ndarray,
    bg_blurred: np.ndarray,
    lut_table: np.ndarray = None,
    blur_ksize: tuple = (17, 17),
    threshold_value: int = 60,
    close_ksize: int = 51,
    padding_pct: float = 0.05,
    edge_margin: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Finds the object bounding box and crops it.
    
    Returns (cropped_224, bbox) where bbox is (x1, y1, x2, y2).
    If no object or touching frame edge, returns (None, None).
    """
    if lut_table is None:
        lut_table = create_gamma_lut(gamma=1.5)

    H, W = img.shape[:2]

    # 1. Blur and background subtraction
    img_blurred = cv2.blur(img, blur_ksize)
    diff = 3*cv2.absdiff(img_blurred, bg_blurred)

    # 2. Extract V channel & apply gamma suppression
    v_channel = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)[:, :, 2]
    v_suppressed = cv2.LUT(v_channel, lut_table)

    # 3. Binary Threshold & Morphological Close
    _, binary = cv2.threshold(v_suppressed, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    mask_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 4. Find contours
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > (H*W/100):
            valid_contours.append(contour)

    if not valid_contours:
        return None, None

    # Convex hull of all foreground contours to get single cohesive bounding box
    all_pts = np.vstack(valid_contours)
    hull = cv2.convexHull(all_pts)
    x, y, bw, bh = cv2.boundingRect(hull)

    # 5. STRICT EDGE & MINIMUM SIZE CHECK
    # a) If box is smaller than 1/13 of image dimensions, treat as noise and skip!
    if bw < (W / 13) or bh < (H / 13):
        return None, None

    # b) If box touches any of the 4 borders, skip!
    if (
        x <= edge_margin or 
        y <= edge_margin or 
        (x + bw) >= (W - edge_margin) or 
        (y + bh) >= (H - edge_margin)
    ):
        return None, None

    # 6. Apply dynamic 5% margin padding
    pad_x = int(min(x, W - (x + bw)) * padding_pct)
    pad_y = int(min(y, H - (y + bh)) * padding_pct)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + bw + pad_x)
    y2 = min(H, y + bh + pad_y)

    crop = img[y1:y2, x1:x2]

    # 7. Resize larger side to 224, enforce even smaller side, and pad to 224x224
    padded_crop = resize_and_pad_to_224(crop, target_size=224)
    return padded_crop, (x1, y1, x2, y2)
