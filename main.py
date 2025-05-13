import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import time
import logging
import signal
import configparser
import threading
import asyncio
import mediapipe as mp

from emotion_recognition import analyze_emotion_async
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.lip_sync import calculate_lip_sync_value
from utils.calculations import fps_calculation
from filter.kalman_filter import KalmanFilter
from utils.shared_variables import SharedVariables
from websocket_client import start_websocket

# Global flag for graceful shutdown
exit_event = threading.Event()

def signal_handler(sig, frame):
    logging.info("Signal received. Exiting gracefully...")
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class FrameProcessor:
    def __init__(self, config, shared_vars: SharedVariables, data_lock: threading.Lock) -> None:
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        # Config thresholds
        self.ear_threshold = config.getfloat("Thresholds", "EAR_THRESHOLD", fallback=0.22)
        self.mar_threshold = config.getfloat("Thresholds", "MAR_THRESHOLD", fallback=0.5)

        self.emotion_analysis_interval = config.getint("Advanced", "EMOTION_ANALYSIS_INTERVAL", fallback=15)
        self.run_emotion = config.getboolean("Advanced", "RUN_EMOTION_ANALYSIS", fallback=True)
        self.emotion_update_delay = config.getfloat("Advanced", "EMOTION_UPDATE_DELAY", fallback=3.0)
        self.last_emotion_time = 0.0

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.ear_kalman = KalmanFilter()
        self.mar_kalman = KalmanFilter()
        self.lip_sync_kalman = KalmanFilter()

        self.frame_count = 0
        self.start_time = time.time()

    def process_landmarks(self, face_landmarks, width: int, height: int):
        left_indices = [33, 160, 158, 133, 153, 144]
        right_indices = [362, 385, 387, 263, 373, 380]
        lip_sync_indices = [13, 14, 61, 291]

        raw_ear_left = calculate_ear([face_landmarks.landmark[i] for i in left_indices], width, height)
        raw_ear_right = calculate_ear([face_landmarks.landmark[i] for i in right_indices], width, height)
        raw_mar = calculate_mar(face_landmarks.landmark, width, height)
        raw_lip_sync = calculate_lip_sync_value([face_landmarks.landmark[i] for i in lip_sync_indices], width, height)

        ear_left = self.ear_kalman.update(raw_ear_left)
        ear_right = self.ear_kalman.update(raw_ear_right)
        mar = self.mar_kalman.update(raw_mar)
        lip_sync_value = self.lip_sync_kalman.update(raw_lip_sync)

        avg_ear = (ear_left + ear_right) / 2
        eye_blinked = raw_ear_left < self.ear_threshold and raw_ear_right < self.ear_threshold
        mouth_open = mar > self.mar_threshold
        lip_sync_active = lip_sync_value > 0.12

        return eye_blinked, mouth_open, lip_sync_active, ear_left, ear_right, mar, lip_sync_value

    def render_overlay(self, frame, eye_status, mouth_status, lip_status, emotion_text, fps):
        overlay_texts = [
            (f"Eye Blinked: {'Yes' if eye_status else 'No'}", (10, 250)),
            (f"Mouth Open: {'Yes' if mouth_status else 'No'}", (10, 280)),
            (f"Lip Sync Active: {'Yes' if lip_status else 'No'}", (10, 310)),
            (f"Emotion: {emotion_text}", (10, 340)),
            (f"FPS: {fps:.2f}", (500, 30))
        ]
        for text, position in overlay_texts:
            font_scale = 0.7 if text.startswith(("Eye", "Mouth", "Lip", "Emotion")) else 0.6
            color = (255, 255, 255) if text.startswith(("Eye", "Mouth", "Lip", "Emotion")) else (0, 255, 0)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return frame

    def process(self, cap: cv2.VideoCapture) -> None:
        while cap.isOpened() and not exit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame. Retrying...")
                continue

            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            eye_blinked = False
            mouth_open = False
            lip_sync_active = False
            emotion_text = "Neutral"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    eye_blinked, mouth_open, lip_sync_active, ear_left, ear_right, mar, lip_sync_value = self.process_landmarks(face_landmarks, width, height)

                    landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                    x1, y1 = np.min(landmarks_array, axis=0).astype(int)
                    x2, y2 = np.max(landmarks_array, axis=0).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    face_roi = frame[y1:y2, x1:x2]

                    current_time = time.time()
                    if self.run_emotion and (current_time - self.last_emotion_time) >= self.emotion_update_delay:
                        try:
                            emotion_text = asyncio.run(analyze_emotion_async(face_roi))  # Perform async emotion analysis
                        except Exception as e:
                            logging.error("Error in async emotion analysis: %s", e, exc_info=True)
                            emotion_text = "Neutral"
                        self.last_emotion_time = current_time

                    with self.data_lock:
                        self.shared_vars.ear_left = ear_left
                        self.shared_vars.ear_right = ear_right
                        self.shared_vars.mar = mar
                        self.shared_vars.lip_sync_value = lip_sync_value

            self.frame_count, fps = fps_calculation(self.frame_count, self.start_time)
            frame = self.render_overlay(frame, eye_blinked, mouth_open, lip_sync_active, emotion_text, fps)
            cv2.imshow("Facial Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Facial Tracker.")

    config = configparser.ConfigParser()
    if not config.read("config.ini"):
        logging.error("Config file missing")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera.")
        return

    shared_vars = SharedVariables()
    data_lock = threading.Lock()

    threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True).start()

    processor = FrameProcessor(config, shared_vars, data_lock)
    processor.process(cap)

if __name__ == "__main__":
    main()
