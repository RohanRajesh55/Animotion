import logging
from typing import List, Tuple, Any
from utils.calculations import calculate_distance_coords

logger = logging.getLogger(__name__)

def calculate_ebr(eyebrow_landmarks: List[Any], eye_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Eyebrow Raise Ratio (EBR) given eyebrow and eye landmarks.

    The EBR is computed by comparing the vertical distance between the central points
    of the eyebrow and the corresponding eye with the width of the eye. Specifically, 
    it is defined as:

        EBR = (vertical_distance between central eyebrow and eye points) / (eye width)

    where:
      - vertical_distance is the Euclidean distance between the central landmark of the eyebrow
        and the corresponding landmark of the eye.
      - eye_width is the horizontal distance between two reference points on the eye (e.g., left-most and right-most landmarks).

    :param eyebrow_landmarks: List of eyebrow landmarks; expected to contain at least 2 points.
    :param eye_landmarks: List of eye landmarks; expected to contain at least 4 points.
    :param width: Width of the frame (in pixels).
    :param height: Height of the frame (in pixels).
    :return: Computed Eyebrow Raise Ratio (float). Returns 0 if the eye width is 0.
    :raises ValueError: If the provided landmark lists do not meet expected lengths.
    """
    # Validate input landmark list lengths.
    if len(eyebrow_landmarks) < 2:
        raise ValueError(f"Expected at least 2 eyebrow landmarks, but got {len(eyebrow_landmarks)}")
    if len(eye_landmarks) < 4:
        raise ValueError(f"Expected at least 4 eye landmarks, but got {len(eye_landmarks)}")
    
    try:
        # Select central points for both the eyebrow and the eye.
        eyebrow_point = eyebrow_landmarks[1]
        eye_point = eye_landmarks[1]
        # Convert normalized landmarks to pixel coordinates.
        eyebrow_coord: Tuple[int, int] = (int(eyebrow_point.x * width), int(eyebrow_point.y * height))
        eye_coord: Tuple[int, int] = (int(eye_point.x * width), int(eye_point.y * height))
    except AttributeError as e:
        logger.error("One or more eyebrow landmarks are missing 'x' and 'y' attributes.", exc_info=e)
        raise

    # Calculate vertical distance between selected eyebrow and eye points.
    vertical_distance = calculate_distance_coords(eyebrow_coord, eye_coord)

    try:
        # Convert reference eye landmarks to pixel coordinates for computing eye width.
        left_eye_coord: Tuple[int, int] = (int(eye_landmarks[0].x * width), int(eye_landmarks[0].y * height))
        right_eye_coord: Tuple[int, int] = (int(eye_landmarks[3].x * width), int(eye_landmarks[3].y * height))
    except AttributeError as e:
        logger.error("One or more eye landmarks are missing 'x' and 'y' attributes.", exc_info=e)
        raise

    # Calculate horizontal distance (eye width) from the designated left and right eye landmarks.
    eye_width = calculate_distance_coords(left_eye_coord, right_eye_coord)
    if eye_width == 0:
        logger.warning("Calculated eye width is zero; returning an EBR of 0.0")
        return 0.0

    ebr = vertical_distance / eye_width
    return ebr