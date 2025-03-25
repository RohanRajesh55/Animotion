# detectors/lip_sync.py

from utils.calculations import calculate_distance_coords

def calculate_lip_sync_value(mouth_landmarks, width, height):
    """
    Calculate Lip Sync Value based on mouth openness and shape.

    :param mouth_landmarks: List of mouth landmarks
    :param width: Width of the frame
    :param height: Height of the frame
    :return: Lip sync value between 0.0 and 1.0
    """
    if len(mouth_landmarks) < 7:
        return 0.0  # Ensure we have enough landmarks

    # Convert landmarks to coordinates
    upper_inner_lip = (int(mouth_landmarks[13].x * width), int(mouth_landmarks[13].y * height))
    lower_inner_lip = (int(mouth_landmarks[14].x * width), int(mouth_landmarks[14].y * height))
    upper_outer_lip = (int(mouth_landmarks[3].x * width), int(mouth_landmarks[3].y * height))
    lower_outer_lip = (int(mouth_landmarks[4].x * width), int(mouth_landmarks[4].y * height))
    left_corner = (int(mouth_landmarks[0].x * width), int(mouth_landmarks[0].y * height))
    right_corner = (int(mouth_landmarks[6].x * width), int(mouth_landmarks[6].y * height))

    # Compute distances
    vertical_inner = calculate_distance_coords(upper_inner_lip, lower_inner_lip)
    vertical_outer = calculate_distance_coords(upper_outer_lip, lower_outer_lip)
    horizontal_distance = calculate_distance_coords(left_corner, right_corner)

    # Prevent divide-by-zero issues
    if horizontal_distance == 0:
        return 0.0

    # Compute weighted lip sync value (smooths animation)
    lip_sync_value = ((0.6 * vertical_inner) + (0.4 * vertical_outer)) / horizontal_distance
    lip_sync_value = min(max(lip_sync_value, 0.0), 1.0)  # Clamp value between 0 and 1

    return lip_sync_value
