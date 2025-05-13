"""
Eyebrow Raise Ratio (EBR).

EBR is defined as the vertical distance between a central eyebrow landmark and the corresponding eye landmark,
normalized by the horizontal eye width.

This version uses subpixel accuracy, error handling, and clean logging.
"""

import logging
from typing import List, Any
import numpy as np
from utils.calculations import calculate_distance_coords

logger = logging.getLogger(__name__)

def calculate_ebr(eyebrow_landmarks: List[Any], eye_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Eyebrow Raise Ratio (EBR) from eyebrow and eye landmarks.

    Args:
        eyebrow_landmarks (List[Any]): List of eyebrow points (must include at least 2).
        eye_landmarks (List[Any]): List of eye points (must include at least 4).
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        float: EBR value (typically ~0.2–0.5 for raised brows). Returns 0.0 on error.
    """
    if len(eyebrow_landmarks) < 2:
        logger.error("Insufficient eyebrow landmarks. Need at least 2, got %d.", len(eyebrow_landmarks))
        return 0.0

    if len(eye_landmarks) < 4:
        logger.error("Insufficient eye landmarks. Need at least 4, got %d.", len(eye_landmarks))
        return 0.0

    try:
        # Extract relevant points (optionally allow flexible selection later)
        eyebrow_point = eyebrow_landmarks[1]
        eye_center_point = eye_landmarks[1]
        eye_outer = eye_landmarks[0]
        eye_inner = eye_landmarks[3]

        # Convert to image coordinates with subpixel precision
        eyebrow_coord = np.array([eyebrow_point.x * width, eyebrow_point.y * height])
        eye_coord = np.array([eye_center_point.x * width, eye_center_point.y * height])
        eye_outer_coord = np.array([eye_outer.x * width, eye_outer.y * height])
        eye_inner_coord = np.array([eye_inner.x * width, eye_inner.y * height])

        # Compute vertical distance (eyebrow raise height)
        vertical_distance = np.linalg.norm(eyebrow_coord - eye_coord)

        # Compute horizontal eye width
        eye_width = np.linalg.norm(eye_outer_coord - eye_inner_coord)

        if eye_width < 1e-5:
            logger.warning("Eye width is near zero. Returning EBR=0 to prevent division error.")
            return 0.0

        ebr = vertical_distance / eye_width
        logger.debug("EBR calculated successfully: %.5f", ebr)
        return round(float(ebr), 5)

    except Exception as e:
        logger.exception("Error calculating Eyebrow Raise Ratio (EBR): %s", e)
        return 0.0
