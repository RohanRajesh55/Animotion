import cv2
import logging
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)

def recognize_emotion(frame: np.ndarray) -> str:
    """
    Recognize the dominant emotion in the provided image frame.

    This function converts the given BGR-format image (as received from OpenCV)
    to RGB and runs DeepFace's emotion analysis. It returns the most likely emotion
    (e.g., "happy", "sad", "neutral", etc.). In case of errors or if detection fails,
    it returns "unknown".

    Args:
        frame (np.ndarray): The input image in BGR format.

    Returns:
        str: The dominant emotion determined by the analysis, or "unknown" if analysis fails.
    """
    try:
        # Convert the frame from BGR to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Perform emotion analysis using DeepFace.
        analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        
        # Check if analysis is a list; if so, use the first element.
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        # Extract and return the dominant emotion.
        emotion = analysis.get("dominant_emotion", "unknown")
        return emotion
    except Exception as e:
        logger.exception("Failed to recognize emotion.", exc_info=e)
        return "unknown"