import logging
from typing import List, Any, Tuple
import numpy as np
from utils.calculations import calculate_distance_coords

logger = logging.getLogger(__name__)

def calculate_ear(eye_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) from eye landmarks.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye_landmarks (List[Any]): List of 6+ points with normalized x/y attributes (float [0,1]).
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        float: EAR value (0.0 to ~0.4 typically). 0.0 if invalid input or calculation fails.
    """
    if not eye_landmarks or len(eye_landmarks) < 6:
        logger.error("Insufficient landmarks provided for EAR calculation. Expected at least 6, got %d.",
                     len(eye_landmarks) if eye_landmarks else 0)
        return 0.0

    try:
        coords = []
        for i, point in enumerate(eye_landmarks[:6]):
            if not hasattr(point, 'x') or not hasattr(point, 'y'):
                logger.error("Landmark %d missing 'x' or 'y' attribute.", i)
                return 0.0
            coords.append((int(point.x * width), int(point.y * height)))

        # Convert to NumPy for vectorized distance calculation
        coords = np.array(coords)

        # EAR formula components
        vertical1 = np.linalg.norm(coords[1] - coords[5])
        vertical2 = np.linalg.norm(coords[2] - coords[4])
        horizontal = np.linalg.norm(coords[0] - coords[3])

        if horizontal < 1e-5:
            logger.warning("Horizontal distance is too small or zero. Returning EAR=0.")
            return 0.0

        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return round(float(ear), 5)

    except Exception as e:
        logger.exception("Error computing EAR: %s", e)
        return 0.0
