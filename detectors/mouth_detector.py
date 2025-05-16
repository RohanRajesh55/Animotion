from utils.calculations import calculate_distance_coords

def calculate_mar(mouth_landmarks, width, height):
    """
    Calculate Mouth Aspect Ratio (MAR) for detecting mouth openness.
    
    This function supports two approaches:
    1. A simplified 4-point method (if exactly 4 landmarks are provided):
         - Landmark order:
             index 0: Left mouth corner
             index 1: Upper lip center
             index 2: Right mouth corner
             index 3: Lower lip center
         - MAR = vertical distance between upper and lower lip / horizontal distance between mouth corners.
    2. A weighted method (if 15 or more landmarks are provided):
         - Computes inner and outer vertical distances and uses a weighted average divided by the horizontal distance.
    
    If an unexpected number of landmarks is provided, it returns 0.0.
    
    :param mouth_landmarks: List of mouth landmarks
    :param width: Width of the frame in pixels
    :param height: Height of the frame in pixels
    :return: Normalized MAR value (0.0 to 1.0)
    """
    if len(mouth_landmarks) == 4:
        # Use the simplified 4-point method.
        coords = [(int(point.x * width), int(point.y * height)) for point in mouth_landmarks]
        # For 4 points:
        #   index 0: left mouth corner
        #   index 1: upper lip center
        #   index 2: right mouth corner
        #   index 3: lower lip center
        vertical = calculate_distance_coords(coords[1], coords[3])
        horizontal = calculate_distance_coords(coords[0], coords[2])
        if horizontal == 0:
            return 0.0  # Prevent division by zero.
        mar = vertical / horizontal
        return min(max(mar, 0.0), 1.0)
    
    elif len(mouth_landmarks) >= 15:
        # Use the weighted method as originally implemented.
        coords = [(int(point.x * width), int(point.y * height)) for point in mouth_landmarks]
        outer_vertical = calculate_distance_coords(coords[3], coords[9])   # Outer lip height
        inner_vertical = calculate_distance_coords(coords[13], coords[14]) # Inner lip height
        horizontal = calculate_distance_coords(coords[0], coords[6])       # Mouth width
        if horizontal == 0:
            return 0.0
        mar = ((0.7 * inner_vertical) + (0.3 * outer_vertical)) / horizontal
        return min(max(mar, 0.0), 1.0)
    else:
        # Not enough landmarks provided.
        return 0.0
