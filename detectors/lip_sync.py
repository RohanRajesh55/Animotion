import logging
from typing import List, Tuple, Any
from utils.calculations import calculate_distance_coords

logger = logging.getLogger(__name__)

def calculate_lip_sync_value(lip_landmarks: List[Any], width: int, height: int) -> float:
    """
    Calculate the Lip Sync Value using selected lip landmarks.

    This function evaluates the intensity of lip movements by computing the ratio of the vertical 
    lip gap (inner lip movement) to the horizontal lip span. This ratio is a useful metric for 
    lip synchronization in animated models.

    The function supports two modes based on the number of provided landmarks:

      1. If exactly 4 landmarks are provided:
         - Assumes landmarks[0] and landmarks[1] represent the outer horizontal lip corners.
         - Assumes landmarks[2] and landmarks[3] represent the inner vertical boundaries 
           (upper and lower lip edges).
         - The lip sync value is computed as:
               lip_sync_value = (distance between landmarks[2] and landmarks[3]) / 
                                (distance between landmarks[0] and landmarks[1])
      
      2. If more than 4 landmarks are provided:
         - Computes a more robust measure by first converting all normalized lip coordinates 
           into pixel positions.
         - The horizontal lip span (outer distance) is defined as the difference between the 
           maximum and minimum x-coordinates of all landmarks.
         - The vertical lip gap (inner distance) is defined as the difference between the 
           maximum and minimum y-coordinates of all landmarks.
         - The lip sync value is then computed as:
               lip_sync_value = (vertical lip gap) / (horizontal lip span)

    The landmarks are assumed to be normalized (values ranging from 0 to 1) and are scaled 
    to the actual frame dimensions using the provided width and height.

    :param lip_landmarks: List of at least 4 landmarks, where each landmark possesses 'x' and 'y' attributes.
    :param width: Width of the frame in pixels.
    :param height: Height of the frame in pixels.
    :return: A float representing the lip sync value.
    :raises ValueError: If fewer than 4 landmarks are provided.
    """
    if len(lip_landmarks) < 4:
        raise ValueError(f"Expected at least 4 lip landmarks, but got {len(lip_landmarks)}.")

    try:
        # Convert each normalized landmark to pixel coordinates.
        coords: List[Tuple[int, int]] = [
            (int(point.x * width), int(point.y * height)) for point in lip_landmarks
        ]
    except AttributeError as e:
        logger.error("One or more lip landmarks are missing the required 'x' and 'y' attributes.", exc_info=e)
        raise

    if len(lip_landmarks) == 4:
        # Use specified landmarks as provided:
        # landmarks[0] and landmarks[1]: Outer lip corners
        # landmarks[2] and landmarks[3]: Inner lip boundaries (upper and lower lips)
        outer_distance = calculate_distance_coords(coords[0], coords[1])
        inner_distance = calculate_distance_coords(coords[2], coords[3])
    else:
        # For more than 4 landmarks, compute a robust measure.
        # Outer lip span is the horizontal bounding box size.
        # Inner lip gap is the vertical bounding box size.
        x_values = [coord[0] for coord in coords]
        y_values = [coord[1] for coord in coords]
        outer_distance = max(x_values) - min(x_values)
        inner_distance = max(y_values) - min(y_values)

    if outer_distance == 0:
        logger.warning("Computed outer lip distance is zero; returning a lip sync value of 0.0 to avoid division by zero.")
        return 0.0

    lip_sync_value = inner_distance / outer_distance
    return lip_sync_value