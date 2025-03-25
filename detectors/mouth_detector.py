import numpy as np

def calculate_mar(mouth_landmarks, width, height):
    """Calculates the Mouth Aspect Ratio (MAR) to detect mouth openness."""
    p1, p2, p3, p4 = mouth_landmarks
    
    vertical = np.linalg.norm(np.array([p2.x * width, p2.y * height]) - np.array([p4.x * width, p4.y * height]))
    horizontal = np.linalg.norm(np.array([p1.x * width, p1.y * height]) - np.array([p3.x * width, p3.y * height]))
    
    mar = vertical / horizontal
    return mar

def detect_speech_activity(mar, threshold=0.6):
    """Detects if the mouth movement is indicative of speech activity."""
    return "Speaking" if mar > threshold else "Silent"
