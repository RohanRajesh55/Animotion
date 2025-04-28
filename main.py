import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import configparser
import logging
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.lip_sync import calculate_lip_sync_value
from websocket_client import start_websocket
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables
from filter.kalman_filter import KalmanFilter
from deepface import DeepFace  # Import DeepFace

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Set up logging
log_level = config.get('Logging', 'LOG_LEVEL', fallback='INFO').upper()
numeric_level = getattr(logging, log_level, logging.INFO)
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Retrieve configuration parameters
DROIDCAM_URL = config.get('Camera', 'DROIDCAM_URL', fallback='0')
EAR_THRESHOLD = config.getfloat('Thresholds', 'EAR_THRESHOLD', fallback=0.22)
MAR_THRESHOLD = config.getfloat('Thresholds', 'MAR_THRESHOLD', fallback=0.5)
EBR_THRESHOLD = config.getfloat('Thresholds', 'EBR_THRESHOLD', fallback=1.5)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start webcam feed
cap = cv2.VideoCapture(DROIDCAM_URL if DROIDCAM_URL != '0' else 0)
if not cap.isOpened():
    logger.error(f"Cannot open camera '{DROIDCAM_URL}'")
    exit(1)

# FPS Calculation
frame_count = 0
start_time = time.time()

data_lock = threading.Lock()
shared_vars = SharedVariables()

# Start WebSocket thread
websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
websocket_thread.start()

# Rolling history for adaptive thresholds
ear_history = []
lip_history = []

# Rolling history for emotion smoothing
emotion_history = []

def smooth_emotion(emotion_text):
    """Smooth the emotion output using a rolling window."""
    if len(emotion_history) > 10:
        emotion_history.pop(0)
    emotion_history.append(emotion_text)
    return max(set(emotion_history), key=emotion_history.count)

def process_frames():
    global frame_count, start_time
    dist_coeffs = np.zeros((4, 1))

    # Initialize Kalman Filters
    ear_kalman = KalmanFilter()
    mar_kalman = KalmanFilter()
    lip_sync_kalman = KalmanFilter()

    while cap.isOpened():
        eye_blinked = "No"
        mouth_open = "No"
        lip_sync_active = "No"
        emotion_text = "None"

        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Retrying...")
            continue

        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                raw_ear_left = calculate_ear([face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]], width, height)
                raw_ear_right = calculate_ear([face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]], width, height)
                raw_mar = calculate_mar([face_landmarks.landmark[i] for i in [61, 291, 13, 14]], width, height)
                raw_lip_sync = calculate_lip_sync_value([face_landmarks.landmark[i] for i in [13, 14, 61, 291]], width, height)

                # Kalman filter updates
                ear_left = ear_kalman.update(raw_ear_left)
                ear_right = ear_kalman.update(raw_ear_right)
                mar = mar_kalman.update(raw_mar)
                lip_sync_value = lip_sync_kalman.update(raw_lip_sync)

                avg_ear = (ear_left + ear_right) / 2
                ear_history.append(avg_ear)
                if len(ear_history) > 50:
                    ear_history.pop(0)
                adaptive_ear_threshold = max(min(ear_history) * 0.85, 0.18)
                eye_blinked = "Yes" if raw_ear_left < adaptive_ear_threshold and raw_ear_right < adaptive_ear_threshold else "No"

                lip_history.append(lip_sync_value)
                if len(lip_history) > 30:
                    lip_history.pop(0)
                smoothed_lip_sync = sum(lip_history[-10:]) / min(len(lip_history), 10)
                lip_sync_active = "Yes" if smoothed_lip_sync > 0.12 else "No"

                mouth_open = "Yes" if raw_mar > MAR_THRESHOLD else "No"

                # Get bounding box of the face
                face_landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                x1, y1 = np.min(face_landmarks_array, axis=0).astype(int)
                x2, y2 = np.max(face_landmarks_array, axis=0).astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                face_roi = frame[y1:y2, x1:x2]

                try:
                    # Perform emotion analysis
                    if face_roi.size != 0:
                        demography = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=True, silent=True)
                        if demography and 'dominant_emotion' in demography[0]:
                            emotion_text = demography[0]['dominant_emotion']
                        else:
                            emotion_text = "No Emotion"
                    else:
                        emotion_text = "No Face"
                except Exception as e:
                    logger.error(f"Emotion detection error: {e}")
                    emotion_text = "Error"

                # Smooth the emotion
                emotion_text = smooth_emotion(emotion_text)

        frame_count, fps = fps_calculation(frame_count, start_time)
        cv2.putText(frame, f"Eye Blinked: {eye_blinked}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouth Open: {mouth_open}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Lip Sync Active: {lip_sync_active}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion_text}", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Facial Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Program terminated.")


if __name__ == "__main__":
    process_frames()
