# detectors/mouth_detector.py

from utils.calculations import calculate_distance_coords

def calculate_mar(mouth_landmarks, width, height):
    """
    Calculate Mouth Aspect Ratio (MAR)

    :param mouth_landmarks: List of mouth landmarks
    :param width: Width of the frame
    :param height: Height of the frame
    :return: Mouth aspect ratio
    """
    # Convert landmarks to 2D coordinates
    coords = [(int(point.x * width), int(point.y * height)) for point in mouth_landmarks]

    # Calculate distances
    vertical = calculate_distance_coords(coords[2], coords[3])
    horizontal = calculate_distance_coords(coords[0], coords[1])

    # Calculate MAR
    mar = vertical / horizontal if horizontal else 0

    return mar
