import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import configparser
import logging
from detectors.eye_detector import EyeDetector
from detectors.mouth_detector import calculate_mar
from detectors.lip_sync import calculate_lip_sync_value
from websocket_client import start_websocket
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

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

# Convert camera input correctly
try:
    DROIDCAM_URL = int(DROIDCAM_URL)
except ValueError:
    pass  # Keep as string if not an integer

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(DROIDCAM_URL)
if not cap.isOpened():
    logger.error(f"Cannot open camera '{DROIDCAM_URL}'")
    exit(1)

# Initialize EyeDetector
eye_detector = EyeDetector()

# FPS Calculation
frame_count = 0
start_time = time.time()
frame_skip = 2  # Process every 2nd frame

data_lock = threading.Lock()
shared_vars = SharedVariables()

# Start WebSocket thread
websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
websocket_thread.start()

# Rolling history for adaptive thresholds
ear_history = []
lip_history = []

def process_frames():
    global frame_count, start_time
    
    while cap.isOpened():
        eye_blinked = "No"
        mouth_open = "No"
        lip_sync_active = "No"

        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Retrying...")
            continue

        # Frame skipping logic
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue

        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                try:
                    # Ensure landmarks exist before calculations
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                    mouth_landmarks = [face_landmarks.landmark[i] for i in [61, 291, 13, 14]]

                    if len(left_eye_landmarks) < 6 or len(right_eye_landmarks) < 6 or len(mouth_landmarks) < 4:
                        continue  # Skip frame if landmarks are incomplete

                    # Eye Aspect Ratio (EAR)
                    ear_left = eye_detector.calculate_ear(left_eye_landmarks, width, height)
                    ear_right = eye_detector.calculate_ear(right_eye_landmarks, width, height)
                    avg_ear = (ear_left + ear_right) / 2

                    # Mouth Aspect Ratio (MAR)
                    mar = calculate_mar(mouth_landmarks, width, height)

                    # Lip Sync Detection
                    lip_sync_value = calculate_lip_sync_value(mouth_landmarks, width, height)

                    # Adaptive EAR threshold
                    ear_history.append(avg_ear)
                    if len(ear_history) > 50:
                        ear_history.pop(0)
                    adaptive_ear_threshold = max(min(ear_history) * 0.85, 0.18)

                    # Adaptive Lip Sync detection
                    lip_history.append(lip_sync_value)
                    if len(lip_history) > 30:
                        lip_history.pop(0)
                    smoothed_lip_sync = sum(lip_history[-10:]) / min(len(lip_history), 10)

                    # Update status
                    eye_blinked = "Yes" if avg_ear < adaptive_ear_threshold else "No"
                    lip_sync_active = "Yes" if smoothed_lip_sync > 0.12 else "No"
                    mouth_open = "Yes" if mar > MAR_THRESHOLD else "No"

                    logger.info(f"EAR: {avg_ear:.2f}, MAR: {mar:.2f}, Lip Sync: {smoothed_lip_sync:.2f}")

                except Exception as e:
                    logger.error(f"Error in landmark processing: {str(e)}")
                    continue

        # FPS Calculation
        frame_count, fps = fps_calculation(frame_count, start_time)

        # Display text on frame
        cv2.putText(frame, f"Eye Blinked: {eye_blinked}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mouth Open: {mouth_open}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Lip Sync Active: {lip_sync_active}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show output
        cv2.imshow('Facial Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Program terminated.")

if __name__ == "__main__":
    try:
        process_frames()
    finally:
        logger.info("Shutting down WebSocket...")
        shared_vars.stop_websocket()
        if data_lock.locked():
            data_lock.release()
        websocket_thread.join()
        logger.info("Websocket thread terminated.")
        time.sleep(1)  # Allow last messages to be sent before exiting
        exit(0)
