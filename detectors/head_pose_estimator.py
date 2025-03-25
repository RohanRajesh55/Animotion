import numpy as np
import cv2

def get_head_pose(face_landmarks, width, height, camera_matrix, dist_coeffs):
    """Calculates head pose estimation using facial landmarks."""
    
    # Define 3D model points (reference facial landmarks)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        (225.0, 170.0, -135.0),      # Right eye corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # 2D facial landmarks from MediaPipe
    image_points = np.array([
        (face_landmarks.landmark[1].x * width, face_landmarks.landmark[1].y * height),  # Nose tip
        (face_landmarks.landmark[199].x * width, face_landmarks.landmark[199].y * height),  # Chin
        (face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height),  # Left eye
        (face_landmarks.landmark[263].x * width, face_landmarks.landmark[263].y * height),  # Right eye
        (face_landmarks.landmark[61].x * width, face_landmarks.landmark[61].y * height),  # Left mouth corner
        (face_landmarks.landmark[291].x * width, face_landmarks.landmark[291].y * height)  # Right mouth corner
    ], dtype=np.float64)
    
    # Solve PnP to get rotation vector
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return "Neutral"
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles (Pitch, Yaw, Roll) - FINAL FIX ✅
    euler_angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    yaw, pitch, roll = euler_angles
    
    # Classify head movement
    if yaw > 10:
        return "Looking Right"
    elif yaw < -10:
        return "Looking Left"
    elif pitch > 10:
        return "Looking Down"
    elif pitch < -10:
        return "Looking Up"
    else:
        return "Neutral"
