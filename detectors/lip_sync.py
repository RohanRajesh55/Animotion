import numpy as np

def calculate_lip_sync_value(mouth_landmarks, width, height):
    """Calculates lip sync value based on mouth movement."""
    p1, p2, p3, p4 = mouth_landmarks
    
    vertical_distance = np.linalg.norm(np.array([p2.x * width, p2.y * height]) - np.array([p4.x * width, p4.y * height]))
    horizontal_distance = np.linalg.norm(np.array([p1.x * width, p1.y * height]) - np.array([p3.x * width, p3.y * height]))
    
    lip_sync_value = vertical_distance / horizontal_distance
    return lip_sync_value

def detect_lip_movement(lip_sync_value, threshold=0.15, history=[]):
    """Detects lip movement using adaptive smoothing."""
    history.append(lip_sync_value)
    if len(history) > 10:
        history.pop(0)
    
    smoothed_value = sum(history) / len(history)
    return "Speaking" if smoothed_value > threshold else "Silent"