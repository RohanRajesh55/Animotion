from typing import List, Tuple, Any
from utils.calculations import calculate_distance_coords

def calculate_ear(eye_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) using six eye landmarks.

    The calculation follows the standard formula:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    where:
      - p1, p2, p3, p4, p5, p6 are the specific landmarks of the eye.
      - The landmarks are assumed to be normalized (values from 0 to 1) and are scaled
        to the actual frame dimensions using the provided width and height.

    :param eye_landmarks: List of exactly 6 landmarks; each should have 'x' and 'y' attributes.
    :param width: Width of the frame (in pixels).
    :param height: Height of the frame (in pixels).
    :return: Computed Eye Aspect Ratio (EAR) as a float.
    :raises ValueError: If the number of provided landmarks is not 6.
    """
    if len(eye_landmarks) != 6:
        raise ValueError(f"Expected exactly 6 eye landmarks, but got {len(eye_landmarks)}")

    # Convert each normalized landmark to pixel coordinates.
    coords: List[Tuple[int, int]] = [
        (int(point.x * width), int(point.y * height)) for point in eye_landmarks
    ]

    # Calculate vertical distances between the landmarks.
    vertical1 = calculate_distance_coords(coords[1], coords[5])
    vertical2 = calculate_distance_coords(coords[2], coords[4])

    # Calculate horizontal distance between the landmarks.
    horizontal = calculate_distance_coords(coords[0], coords[3])

    # Prevent division by zero in case of an unexpected scenario.
    if horizontal == 0:
        return 0.0

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear
