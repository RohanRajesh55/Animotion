import cv2
import numpy as np

class FacePreprocessor:
    """
    Preprocesses face images for MobileNetV2 and ResEmoNet models.
    This includes:
    - Extracting face ROI from landmark coordinates
    - Resizing to model-specific input size
    - Normalization for consistent model input
    """

    def __init__(self, model_input_size=(224, 224)):
        """
        Initialize the face preprocessor.

        :param model_input_size: Tuple (width, height) for resizing images
        """
        self.model_input_size = model_input_size

    def extract_face_roi(self, frame, face_landmarks, margin=20):
        """
        Extracts the face region of interest (ROI) from the given frame.

        :param frame: The input image frame (BGR)
        :param face_landmarks: List of facial landmarks (MediaPipe or OpenCV format)
        :param margin: Additional margin around the detected face (in pixels)
        :return: Cropped face ROI or None if extraction fails
        """
        if not face_landmarks or len(face_landmarks) < 2:
            return None  # No valid face detected

        # Get bounding box around the face
        x_min = min(int(landmark.x * frame.shape[1]) for landmark in face_landmarks)
        y_min = min(int(landmark.y * frame.shape[0]) for landmark in face_landmarks)
        x_max = max(int(landmark.x * frame.shape[1]) for landmark in face_landmarks)
        y_max = max(int(landmark.y * frame.shape[0]) for landmark in face_landmarks)

        # Expand the bounding box by the margin
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, frame.shape[1])
        y_max = min(y_max + margin, frame.shape[0])

        # Extract the face ROI
        face_roi = frame[y_min:y_max, x_min:x_max]
        if face_roi.size == 0:
            return None  # Return None if extraction failed

        return face_roi

    def preprocess_for_model(self, face_roi):
        """
        Preprocesses the extracted face ROI for MobileNetV2 and ResEmoNet models.

        :param face_roi: Cropped face image
        :return: Preprocessed image ready for model inference
        """
        if face_roi is None:
            return None

        # Convert to RGB (OpenCV loads images in BGR)
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        face_resized = cv2.resize(face_rgb, self.model_input_size)

        # Normalize pixel values (0 to 1)
        face_normalized = face_resized.astype(np.float32) / 255.0

        # Convert to tensor format (H, W, C) -> (1, H, W, C)
        face_tensor = np.expand_dims(face_normalized, axis=0)

        return face_tensor
