# detectors/facial_landmarks_processor.py

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Optional, Dict, Any
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.eyebrow_detector import calculate_ebr
from detectors.head_pose_estimator import get_head_pose
from detectors.lip_sync import calculate_lip_sync_value

logger = logging.getLogger(__name__)

class FacialLandmarksProcessor:
    def __init__(self, max_num_faces: int = 1) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process an input BGR frame to extract facial metrics.
        
        Metrics computed:
          - Left and Right Eye Aspect Ratio (EAR)
          - Mouth Aspect Ratio (MAR)
          - Left and Right Eyebrow Raise Ratio (EBR)
          - Lip Sync Value
          - Head Pose (rotation and translation vectors)
        
        Args:
            frame (np.ndarray): Input image frame (BGR format).
        
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing computed metrics, or None if no face is detected.
        """
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            logger.debug("No face detected in the frame.")
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Compute EAR for both eyes.
        try:
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            ear_left = calculate_ear(left_eye_landmarks, width, height)
            ear_right = calculate_ear(right_eye_landmarks, width, height)
        except Exception as e:
            logger.error("Error computing EAR: %s", e)
            ear_left = ear_right = None
        
        # Compute MAR using 4 specific mouth landmarks.
        try:
            mouth_landmarks = [face_landmarks.landmark[i] for i in [61, 291, 13, 14]]
            mar = calculate_mar(mouth_landmarks, width, height)
        except Exception as e:
            logger.error("Error computing MAR: %s", e)
            mar = None
        
        # Compute EBR for left eyebrow.
        try:
            # Example indices for left eyebrow landmarks (adjust as per calibration).
            left_eyebrow_landmarks = [face_landmarks.landmark[i] for i in [70, 63]]
            ebr_left = calculate_ebr(left_eyebrow_landmarks, left_eye_landmarks, width, height)
        except Exception as e:
            logger.error("Error computing left EBR: %s", e)
            ebr_left = None
        
        # Compute EBR for right eyebrow.
        try:
            # Example indices for right eyebrow landmarks.
            right_eyebrow_landmarks = [face_landmarks.landmark[i] for i in [334, 296]]
            ebr_right = calculate_ebr(right_eyebrow_landmarks, right_eye_landmarks, width, height)
        except Exception as e:
            logger.error("Error computing right EBR: %s", e)
            ebr_right = None
        
        # Compute Lip Sync Value using mouth landmarks.
        try:
            lip_sync_value = calculate_lip_sync_value(mouth_landmarks, width, height)
        except Exception as e:
            logger.error("Error computing lip sync value: %s", e)
            lip_sync_value = None
        
        # Compute Head Pose using 6 facial landmarks.
        try:
            image_points = np.array([
                [face_landmarks.landmark[1].x * width, face_landmarks.landmark[1].y * height],     # Nose tip (example)
                [face_landmarks.landmark[152].x * width, face_landmarks.landmark[152].y * height],  # Chin
                [face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height],    # Left eye corner
                [face_landmarks.landmark[263].x * width, face_landmarks.landmark[263].y * height],  # Right eye corner
                [face_landmarks.landmark[61].x * width, face_landmarks.landmark[61].y * height],    # Left mouth corner
                [face_landmarks.landmark[291].x * width, face_landmarks.landmark[291].y * height]   # Right mouth corner
            ], dtype="double")
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion.
            rotation_vector, translation_vector = get_head_pose(image_points, camera_matrix, dist_coeffs)
        except Exception as e:
            logger.error("Error computing head pose: %s", e)
            rotation_vector = translation_vector = None
        
        metrics: Dict[str, Any] = {
            "ear_left": ear_left,
            "ear_right": ear_right,
            "mar": mar,
            "ebr_left": ebr_left,
            "ebr_right": ebr_right,
            "lip_sync_value": lip_sync_value,
            "head_pose": {
                "rotation_vector": rotation_vector,
                "translation_vector": translation_vector
            }
        }
        return metrics

    def __del__(self) -> None:
        self.face_mesh.close()