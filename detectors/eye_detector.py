# detectors/eye_detector.py

from utils.calculations import calculate_distance_coords

def calculate_ear(eye_landmarks, width, height):
    """
    Calculate Eye Aspect Ratio (EAR)

    :param eye_landmarks: List of eye landmarks
    :param width: Width of the frame
    :param height: Height of the frame
    :return: Eye aspect ratio
    """
    # Convert landmarks to 2D coordinates
    coords = [(int(point.x * width), int(point.y * height)) for point in eye_landmarks]

    # Calculate distances between the vertical eye landmarks
    vertical1 = calculate_distance_coords(coords[1], coords[5])
    vertical2 = calculate_distance_coords(coords[2], coords[4])

    # Calculate distance between the horizontal eye landmarks
    horizontal = calculate_distance_coords(coords[0], coords[3])

    # Calculate EAR
    ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0

    return ear
