import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Optional, Dict, Any

# Importing helper functions for computing metrics.
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.eyebrow_detector import calculate_ebr
from detectors.head_pose_estimator import get_head_pose
from detectors.lip_sync import calculate_lip_sync_value

logger = logging.getLogger(__name__)

class FacialLandmarksProcessor:
    """
    A production-level processor that uses MediaPipe Face Mesh to extract facial landmarks 
    and compute various metrics used to drive the animated avatar.
    
    Metrics computed:
      - Left and Right Eye Aspect Ratios (EAR)
      - Mouth Aspect Ratio (MAR)
      - Left and Right Eyebrow Raise Ratio (EBR)
      - Lip Sync Value
      - Head Pose: rotation and translation vectors
    """
    def __init__(self, max_num_faces: int = 1) -> None:
        """
        Initialize the FacialLandmarksProcessor with the specified settings.
        
        Args:
            max_num_faces (int): Maximum number of faces to detect (default 1).
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("Initialized MediaPipe FaceMesh processor.")
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process an input BGR frame to extract and compute facial metrics.
        
        This function converts the input frame to RGB, runs the face detection model,
        and if a face is found, computes:
        
          - Left and Right Eye Aspect Ratio (EAR) using designated eye landmarks.
          - Mouth Aspect Ratio (MAR) using specified mouth landmarks.
          - Eyebrow Raise Ratio (EBR) for left/right eyebrows.
          - Lip Sync Value based on mouth openness.
          - Head Pose based on 6 key facial landmarks.
        
        Args:
            frame (np.ndarray): The input color frame in BGR format.
            
        Returns:
            Optional[Dict[str, Any]]: A dictionary containing computed metrics if a
                                       face is detected; otherwise, None.
        """
        height, width = frame.shape[:2]
        # Convert the frame to RGB as required by MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            logger.debug("No face detected in the frame.")
            return None
        
        # Only process the first detected face.
        face_landmarks = results.multi_face_landmarks[0]
        
        # Initialize dictionary to hold computed metrics.
        metrics: Dict[str, Any] = {}
        
        # Compute Eye Aspect Ratio (EAR) for both eyes.
        try:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
            right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
            ear_left = calculate_ear(left_eye_landmarks, width, height)
            ear_right = calculate_ear(right_eye_landmarks, width, height)
            metrics["ear_left"] = ear_left
            metrics["ear_right"] = ear_right
            logger.debug(f"Computed EAR: left={ear_left:.4f} right={ear_right:.4f}")
        except Exception as e:
            logger.error("Error computing EAR: %s", e)
            metrics["ear_left"] = metrics["ear_right"] = None
        
        # Compute Mouth Aspect Ratio (MAR) using four mouth landmarks.
        try:
            # Provided indices chosen for mouth metrics; adjust after calibration.
            mouth_indices = [61, 291, 13, 14]
            mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]
            mar = calculate_mar(mouth_landmarks, width, height)
            metrics["mar"] = mar
            logger.debug(f"Computed MAR: {mar:.4f}")
        except Exception as e:
            logger.error("Error computing MAR: %s", e)
            metrics["mar"] = None
        
        # Compute Eyebrow Raise Ratio (EBR) for left eyebrow.
        try:
            # Indices for left eyebrow landmarks (example) and corresponding left eye landmarks.
            left_eyebrow_indices = [70, 63]  # Adjust as needed.
            left_eyebrow_landmarks = [face_landmarks.landmark[i] for i in left_eyebrow_indices]
            ebr_left = calculate_ebr(left_eyebrow_landmarks, left_eye_landmarks, width, height)
            metrics["ebr_left"] = ebr_left
            logger.debug(f"Computed left EBR: {ebr_left:.4f}")
        except Exception as e:
            logger.error("Error computing left EBR: %s", e)
            metrics["ebr_left"] = None
        
        # Compute Eyebrow Raise Ratio for right eyebrow.
        try:
            right_eyebrow_indices = [334, 296]  # Adjust indices after calibration.
            right_eyebrow_landmarks = [face_landmarks.landmark[i] for i in right_eyebrow_indices]
            ebr_right = calculate_ebr(right_eyebrow_landmarks, right_eye_landmarks, width, height)
            metrics["ebr_right"] = ebr_right
            logger.debug(f"Computed right EBR: {ebr_right:.4f}")
        except Exception as e:
            logger.error("Error computing right EBR: %s", e)
            metrics["ebr_right"] = None
        
        # Compute Lip Sync Value using the mouth landmarks.
        try:
            lip_sync_value = calculate_lip_sync_value(mouth_landmarks, width, height)
            metrics["lip_sync_value"] = lip_sync_value
            logger.debug(f"Computed lip sync value: {lip_sync_value:.4f}")
        except Exception as e:
            logger.error("Error computing lip sync value: %s", e)
            metrics["lip_sync_value"] = None
        
        # Compute Head Pose using 6 selected facial landmarks.
        try:
            # Example: use landmarks for nose tip, chin, left/right eye corners, and mouth corners.
            image_points = np.array([
                [face_landmarks.landmark[1].x * width, face_landmarks.landmark[1].y * height],     # Nose tip
                [face_landmarks.landmark[152].x * width, face_landmarks.landmark[152].y * height],  # Chin
                [face_landmarks.landmark[33].x * width, face_landmarks.landmark[33].y * height],    # Left eye corner
                [face_landmarks.landmark[263].x * width, face_landmarks.landmark[263].y * height],  # Right eye corner
                [face_landmarks.landmark[61].x * width, face_landmarks.landmark[61].y * height],    # Left mouth corner
                [face_landmarks.landmark[291].x * width, face_landmarks.landmark[291].y * height]   # Right mouth corner
            ], dtype="double")
            
            focal_length = width  # Assuming focal length approximates the width.
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion.
            rotation_vector, translation_vector = get_head_pose(image_points, camera_matrix, dist_coeffs)
            metrics["head_pose"] = {
                "rotation_vector": rotation_vector,
                "translation_vector": translation_vector
            }
            logger.debug("Computed head pose.")
        except Exception as e:
            logger.error("Error computing head pose: %s", e)
            metrics["head_pose"] = {
                "rotation_vector": None,
                "translation_vector": None
            }
        
        return metrics
    
    def __del__(self) -> None:
        """
        Release resources by closing the MediaPipe FaceMesh instance.
        """
        if self.face_mesh:
            self.face_mesh.close()
            logger.info("Released FaceMesh resources.")