import cv2
import numpy as np
import mediapipe as mp
from utils.calculations import calculate_distance_coords
from models.model_utils import preprocess_face

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class FrameProcessor:
    def __init__(self, image_size=(224, 224)):
        """
        Initializes the frame processor.

        :param image_size: Target size for facial image preprocessing
        """
        self.image_size = image_size
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )

    def detect_facial_landmarks(self, frame):
        """
        Detects facial landmarks using MediaPipe FaceMesh.

        :param frame: Input video frame (BGR)
        :return: Tuple (face_landmarks, annotated_frame) or (None, frame) if no face detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0], self.draw_landmarks(frame, results.multi_face_landmarks[0])
        
        return None, frame

    def draw_landmarks(self, frame, face_landmarks):
        """
        Draws facial landmarks on the frame.

        :param frame: Input frame
        :param face_landmarks: Detected landmarks from MediaPipe FaceMesh
        :return: Annotated frame
        """
        mp_drawing.draw_landmarks(
            frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
        )
        return frame

    def extract_face_roi(self, frame, face_landmarks):
        """
        Extracts the face region from the frame using facial landmarks.

        :param frame: Input frame
        :param face_landmarks: Detected facial landmarks
        :return: Cropped face image or None if face detection fails
        """
        h, w, _ = frame.shape
        x_coords = [int(l.x * w) for l in face_landmarks.landmark]
        y_coords = [int(l.y * h) for l in face_landmarks.landmark]

        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

        if (x_max - x_min) > 10 and (y_max - y_min) > 10:
            face_roi = frame[y_min:y_max, x_min:x_max]
            return cv2.resize(face_roi, self.image_size)
        return None

    def process_frame(self, frame):
        """
        Processes a video frame: detects face, extracts ROI, and preprocesses it.

        :param frame: Input video frame (BGR)
        :return: Preprocessed face tensor (or None if no face detected)
        """
        face_landmarks, annotated_frame = self.detect_facial_landmarks(frame)

        if face_landmarks:
            face_roi = self.extract_face_roi(frame, face_landmarks)
            if face_roi is not None:
                return preprocess_face(face_roi), annotated_frame
        
        return None, frame
