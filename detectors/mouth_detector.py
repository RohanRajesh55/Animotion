# detectors/mouth_detector.py

from utils.calculations import calculate_distance_coords

def calculate_mar(mouth_landmarks, width, height):
    """
    Calculate Mouth Aspect Ratio (MAR) for detecting mouth openness.

    :param mouth_landmarks: List of mouth landmarks
    :param width: Width of the frame
    :param height: Height of the frame
    :return: Normalized MAR value (0.0 - 1.0)
    """
    if len(mouth_landmarks) < 15:
        return 0.0  # Ensure we have enough landmarks

    # Convert landmarks to 2D coordinates
    coords = [(int(point.x * width), int(point.y * height)) for point in mouth_landmarks]

    # Compute distances for MAR calculation
    outer_vertical = calculate_distance_coords(coords[3], coords[9])   # Outer lip height
    inner_vertical = calculate_distance_coords(coords[13], coords[14]) # Inner lip height
    horizontal = calculate_distance_coords(coords[0], coords[6])       # Mouth width

    # Prevent divide-by-zero issues
    if horizontal == 0:
        return 0.0

    # Compute MAR with weighted averaging for smoother animation
    mar = ((0.7 * inner_vertical) + (0.3 * outer_vertical)) / horizontal
    mar = min(max(mar, 0.0), 1.0)  # Keep MAR between 0 and 1

    return mar
