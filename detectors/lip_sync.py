# detectors/lip_sync.py

from utils.calculations import calculate_distance_coords

def calculate_lip_sync_value(mouth_landmarks, width, height):
    """
    Calculate Lip Sync Value based on mouth openness and shape

    :param mouth_landmarks: List of mouth landmarks
    :param width: Width of the frame
    :param height: Height of the frame
    :return: Lip sync value between 0.0 and 1.0
    """
    # Convert landmarks to coordinates
    upper_inner_lip = (int(mouth_landmarks[0].x * width), int(mouth_landmarks[0].y * height))
    lower_inner_lip = (int(mouth_landmarks[1].x * width), int(mouth_landmarks[1].y * height))
    left_corner = (int(mouth_landmarks[2].x * width), int(mouth_landmarks[2].y * height))
    right_corner = (int(mouth_landmarks[3].x * width), int(mouth_landmarks[3].y * height))

    # Vertical and horizontal distances
    vertical_distance = calculate_distance_coords(upper_inner_lip, lower_inner_lip)
    horizontal_distance = calculate_distance_coords(left_corner, right_corner)

    # Calculate mouth openness ratio
    mouth_openness = vertical_distance / horizontal_distance if horizontal_distance else 0

    # Map the value to a range suitable for your application (e.g., 0.0 to 1.0)
    lip_sync_value = min(max(mouth_openness, 0.0), 1.0)

    return lip_sync_value
