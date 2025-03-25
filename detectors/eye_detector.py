import numpy as np
import cv2
import mediapipe as mp

def calculate_ear(eye_landmarks, width, height):
    """Calculates the Eye Aspect Ratio (EAR) to detect blinks."""
    p1, p2, p3, p4, p5, p6 = eye_landmarks
    
    vertical_1 = np.linalg.norm(np.array([p2.x * width, p2.y * height]) - np.array([p6.x * width, p6.y * height]))
    vertical_2 = np.linalg.norm(np.array([p3.x * width, p3.y * height]) - np.array([p5.x * width, p5.y * height]))
    horizontal = np.linalg.norm(np.array([p1.x * width, p1.y * height]) - np.array([p4.x * width, p4.y * height]))
    
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def estimate_gaze(face_landmarks, width, height):
    """Estimates gaze direction based on eye landmarks."""
    left_eye_center = np.array([(face_landmarks[33].x + face_landmarks[133].x) / 2 * width, 
                                (face_landmarks[33].y + face_landmarks[133].y) / 2 * height])
    right_eye_center = np.array([(face_landmarks[362].x + face_landmarks[263].x) / 2 * width, 
                                 (face_landmarks[362].y + face_landmarks[263].y) / 2 * height])
    
    nose_tip = np.array([face_landmarks[1].x * width, face_landmarks[1].y * height])
    
    gaze_vector = (left_eye_center + right_eye_center) / 2 - nose_tip
    gaze_direction = "Center"
    
    if gaze_vector[0] < -5:
        gaze_direction = "Left"
    elif gaze_vector[0] > 5:
        gaze_direction = "Right"
    
    return gaze_direction