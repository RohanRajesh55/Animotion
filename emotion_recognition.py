import logging

# Check for GPU availability using TensorFlow.
try:
    import tensorflow as tf
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        logging.info("GPU(s) detected: " + ", ".join([device.name for device in gpu_devices]))
    else:
        logging.info("No GPU detected. Running emotion analysis on CPU.")
except ImportError:
    logging.warning("TensorFlow not installed. Cannot check GPU availability; proceeding with default settings.")

# Attempt to import DeepFace for emotion recognition.
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logging.warning("DeepFace is not installed. Emotion recognition will default to 'Neutral'.")
    DEEPFACE_AVAILABLE = False

def analyze_emotion(face_roi, use_emotion_recognition=True, detector_backend='opencv'):
    """
    Analyze the emotion from the provided face region of interest (ROI) using DeepFace.
    If a GPU is detected (with the GPU-enabled TensorFlow installed), DeepFace will run on the GPU;
    otherwise, the analysis will run on the CPU.

    Args:
        face_roi (numpy.ndarray): The image region containing the face.
        use_emotion_recognition (bool): Whether to enable emotion recognition.
        detector_backend (str): The backend detector to use with DeepFace.

    Returns:
        str: Dominant emotion as detected by DeepFace or 'Neutral' if detection is disabled or fails.
    """
    if use_emotion_recognition and DEEPFACE_AVAILABLE:
        try:
            result = DeepFace.analyze(
                face_roi,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=detector_backend
            )
            # DeepFace might return a list if multiple faces are analyzed;
            # in that case, just use the first result.
            if isinstance(result, list):
                result = result[0]
            return result.get("dominant_emotion", "Neutral")
        except Exception as e:
            logging.error(f"Emotion analysis error: {e}")
            return "Neutral"
    else:
        return "Neutral"