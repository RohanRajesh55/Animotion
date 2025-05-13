"""
lip sync value.

Lip sync value is computed as the ratio of vertical mouth opening to horizontal mouth width,
based on key mouth landmarks. The output is normalized to [0.0, 1.0].
"""

import logging
from typing import List, Any, Tuple
from utils.calculations import calculate_distance_coords

logger = logging.getLogger(__name__)

# Constants for mouth landmark indexing
UPPER_INNER_LIP = 0
LOWER_INNER_LIP = 1
LEFT_CORNER = 2
RIGHT_CORNER = 3

def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Clamp a value to a specified range."""
    return max(min(value, max_value), min_value)

def calculate_lip_sync_value(
    mouth_landmarks: List[Any],
    width: int,
    height: int,
    openness_range: Tuple[float, float] = (0.0, 1.0)
) -> float:
    """
    Calculate Lip Sync Value from mouth landmarks.

    Parameters:
        mouth_landmarks (List[Any]): List of normalized mouth landmarks with at least 4 key points.
        width (int): Width of the image/frame.
        height (int): Height of the image/frame.
        openness_range (Tuple[float, float]): Optional min-max range to normalize openness.

    Returns:
        float: Normalized lip sync value in the range [0.0, 1.0]. Returns 0.0 on error.
    """
    if not mouth_landmarks or len(mouth_landmarks) < 4:
        logger.error("Insufficient mouth landmarks; expected 4, got %d.",
                     len(mouth_landmarks) if mouth_landmarks else 0)
        return 0.0

    try:
        # Convert normalized coordinates to image-space.
        upper = (int(mouth_landmarks[UPPER_INNER_LIP].x * width),
                 int(mouth_landmarks[UPPER_INNER_LIP].y * height))
        lower = (int(mouth_landmarks[LOWER_INNER_LIP].x * width),
                 int(mouth_landmarks[LOWER_INNER_LIP].y * height))
        left = (int(mouth_landmarks[LEFT_CORNER].x * width),
                int(mouth_landmarks[LEFT_CORNER].y * height))
        right = (int(mouth_landmarks[RIGHT_CORNER].x * width),
                 int(mouth_landmarks[RIGHT_CORNER].y * height))

        vertical_dist = calculate_distance_coords(upper, lower)
        horizontal_dist = calculate_distance_coords(left, right)

        if horizontal_dist == 0:
            logger.warning("Horizontal mouth width is zero; cannot compute lip sync value.")
            return 0.0

        raw_openness = vertical_dist / horizontal_dist

        # Normalize and clamp the result
        min_openness, max_openness = openness_range
        normalized = (raw_openness - min_openness) / (max_openness - min_openness)
        lip_sync_value = clamp(normalized)

        return lip_sync_value

    except Exception as e:
        logger.exception("Exception while calculating lip sync value: %s", e)
        return 0.0
