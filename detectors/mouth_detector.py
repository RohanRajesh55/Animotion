# detectors/mouth_detector.py

from utils.calculations import calculate_distance_coords

def calculate_mar(mouth_landmarks, width, height, multiplier=1.0):
    """
    Calculate Mouth Aspect Ratio (MAR) for detecting mouth openness.

    :param mouth_landmarks: List of mouth landmarks
    :param width: Width of the frame
    :param height: Height of the frame
    :param multiplier: Optional multiplier to scale MAR (default = 1.0)
    :return: Normalized MAR value (0.0 - 1.0)
    """
    if len(mouth_landmarks) < 15:
        return 0.0  # Not enough points for a good MAR calculation

    # Convert normalized landmarks to pixel coordinates
    coords = [(int(point.x * width), int(point.y * height)) for point in mouth_landmarks]

    # Compute relevant distances
    outer_vertical = calculate_distance_coords(coords[3], coords[9])   # Upper to lower outer lip
    inner_vertical = calculate_distance_coords(coords[13], coords[14]) # Inner lip height
    horizontal = calculate_distance_coords(coords[0], coords[6])       # Mouth width

    if horizontal == 0:
        return 0.0  # Avoid division by zero

    # MAR formula (weighted average for smoothness)
    mar = ((0.7 * inner_vertical) + (0.3 * outer_vertical)) / horizontal
    mar *= multiplier  # Apply optional multiplier
    mar = min(max(mar, 0.0), 1.0)  # Clamp between 0 and 1

    return mar
