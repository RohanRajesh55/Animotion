import logging
from typing import List, Tuple, Any
from utils.calculations import calculate_distance_coords

logger = logging.getLogger(__name__)

def calculate_smile_intensity(mouth_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate a normalized smile intensity value from mouth landmarks.

    This function computes smile intensity by comparing the vertical position of the 
    mouth corners with that of the upper lip. Specifically, it assumes the following:
      - 'mouth_landmarks[0]' corresponds to the left mouth corner (e.g., index 61).
      - 'mouth_landmarks[1]' corresponds to the upper lip center (e.g., index 13).
      - 'mouth_landmarks[2]' corresponds to the right mouth corner (e.g., index 291).

    The smile intensity is computed as the average vertical difference between the upper lip and 
    each mouth corner, normalized by the horizontal mouth width. A higher positive value indicates a 
    higher upward displacement (i.e. a smile). The metric is unitless and can be further calibrated if needed.

    Args:
        mouth_landmarks (List[Any]): List of at least 3 landmarks with normalized coordinates.
                                     Expected order: [left mouth corner, upper lip center, right mouth corner].
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        float: A normalized smile intensity value. Higher values indicate a stronger smile.
    
    Raises:
        ValueError: If fewer than 3 landmarks are provided.
    """
    if len(mouth_landmarks) < 3:
        raise ValueError(f"Expected at least 3 mouth landmarks (for smile detection), but got {len(mouth_landmarks)}.")

    try:
        # Convert normalized landmarks to pixel coordinates.
        left_corner: Tuple[int, int] = (
            int(mouth_landmarks[0].x * width), int(mouth_landmarks[0].y * height)
        )
        upper_lip: Tuple[int, int] = (
            int(mouth_landmarks[1].x * width), int(mouth_landmarks[1].y * height)
        )
        right_corner: Tuple[int, int] = (
            int(mouth_landmarks[2].x * width), int(mouth_landmarks[2].y * height)
        )
    except AttributeError as e:
        logger.error("One or more mouth landmarks are missing the required 'x' and 'y' attributes.", exc_info=e)
        raise

    # Compute horizontal mouth width (distance between the two mouth corners).
    horizontal_width = calculate_distance_coords(left_corner, right_corner)
    if horizontal_width == 0:
        logger.warning("Horizontal mouth width is zero; returning smile intensity as 0.0.")
        return 0.0

    # Calculate vertical differences: difference between the upper lip and each mouth corner.
    left_vertical_diff = upper_lip[1] - left_corner[1]
    right_vertical_diff = upper_lip[1] - right_corner[1]

    # Average the vertical differences.
    avg_vertical_diff = (left_vertical_diff + right_vertical_diff) / 2.0

    # Normalize by the horizontal width to obtain a relative smile intensity.
    smile_intensity = avg_vertical_diff / horizontal_width

    return smile_intensity