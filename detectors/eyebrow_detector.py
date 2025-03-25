import numpy as np

def calculate_ebr(eyebrow_landmarks, eye_landmarks, width, height):
    """Calculates the Eyebrow Raise Ratio (EBR) to detect eyebrow movement."""
    
    # Get key points
    brow_mid = eyebrow_landmarks[2]
    eye_mid = eye_landmarks[2]
    
    # Vertical distance between eyebrow mid-point and eye mid-point
    vertical_distance = np.linalg.norm(
        np.array([brow_mid.x * width, brow_mid.y * height]) - 
        np.array([eye_mid.x * width, eye_mid.y * height])
    )
    
    return vertical_distance

def detect_eyebrow_gesture(ebr, neutral_threshold=1.2, raised_threshold=1.6):
    """Detects eyebrow gestures based on EBR values."""
    if ebr > raised_threshold:
        return "Raised"
    elif ebr < neutral_threshold:
        return "Lowered"
    else:
        return "Neutral"