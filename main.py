#!/usr/bin/env python3
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import configparser
import logging
import signal
import argparse
import asyncio
from collections import deque

from detectors.eye_detector import calculate_ear
from detectors.lip_sync import calculate_lip_sync_value
from detectors.mouth_detector import calculate_mar
from emotion_detector import predict_emotion
from websocket_client import start_websocket
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables
from filter.kalman_filter import KalmanFilter

# Global event for graceful shutdown.
exit_event = threading.Event()

def signal_handler(sig, frame):
    logging.info("Signal received (%s). Exiting gracefully...", sig)
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Live Face-to-Animation Tracker")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to configuration file")
    return parser.parse_args()

async def process_emotion_async(face_roi):
    from emotion_detector import predict_emotion
    return await asyncio.to_thread(predict_emotion, face_roi)

class FrameProcessor:
    def __init__(self, config, shared_vars: SharedVariables, data_lock: threading.Lock) -> None:
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        self.ear_threshold = config.getfloat("Thresholds", "EAR_THRESHOLD", fallback=0.22)
        self.mar_threshold = config.getfloat("Thresholds", "MAR_THRESHOLD", fallback=0.5)
        self.emotion_update_delay = config.getfloat("Advanced", "EMOTION_UPDATE_DELAY", fallback=3.0)
        self.run_emotion = config.getboolean("Advanced", "RUN_EMOTION_ANALYSIS", fallback=True)
        self.last_emotion_time = 0.0
        self.current_emotion = "Neutral"

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

        self.ear_history = deque(maxlen=50)
        self.lip_history = deque(maxlen=30)
        self.frame_count = 0
        self.start_time = time.time()

    def process_landmarks(self, face_landmarks, width: int, height: int):
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        mouth_indices = [61, 0, 291, 17]

        raw_ear_left = calculate_ear([face_landmarks.landmark[i] for i in left_eye_indices], width, height)
        raw_ear_right = calculate_ear([face_landmarks.landmark[i] for i in right_eye_indices], width, height)
        mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]
        raw_mar = calculate_mar(mouth_landmarks, width, height)
        raw_lip_sync = calculate_lip_sync_value([face_landmarks.landmark[i] for i in [13, 14, 61, 291]], width, height)

        ear_left = self.ear_kalman.update(raw_ear_left)
        ear_right = self.ear_kalman.update(raw_ear_right)
        mar = self.mar_kalman.update(raw_mar)
        lip_sync_value = self.lip_sync_kalman.update(raw_lip_sync)

        avg_ear = (ear_left + ear_right) / 2
        self.ear_history.append(avg_ear)
        adaptive_ear_threshold = max(min(self.ear_history) * 0.85, 0.18)
        eye_blinked = raw_ear_left < adaptive_ear_threshold and raw_ear_right < adaptive_ear_threshold

        self.lip_history.append(lip_sync_value)
        smoothed_lip_sync = np.mean(self.lip_history)
        lip_sync_active = smoothed_lip_sync > 0.12
        mouth_open = mar > self.mar_threshold

        return eye_blinked, mouth_open, lip_sync_active, ear_left, ear_right, mar, lip_sync_value

    def render_overlay(self, frame, eye_status: bool, mouth_status: bool, lip_status: bool, emotion_text: str, fps: float):
        overlays = [
            (f"Eye Blinked: {'Yes' if eye_status else 'No'}", (10, 250)),
            (f"Mouth Open: {'Yes' if mouth_status else 'No'}", (10, 280)),
            (f"Lip Sync: {'Active' if lip_status else 'Inactive'}", (10, 310)),
            (f"Emotion: {emotion_text}", (10, 340)),
            (f"FPS: {fps:.2f}", (500, 30))
        ]
        for text, pos in overlays:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
            emotion_text = self.current_emotion

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    (eye_blinked, mouth_open, lip_sync_active,
                     ear_left, ear_right, mar, lip_sync_value) = self.process_landmarks(face_landmarks, width, height)

                    # Define ROI from landmarks.
                    landmarks_arr = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                    x1, y1 = np.min(landmarks_arr, axis=0).astype(int)
                    x2, y2 = np.max(landmarks_arr, axis=0).astype(int)
                    face_roi = frame[y1:y2, x1:x2]

                    current_time = time.time()
                    if self.run_emotion and (current_time - self.last_emotion_time) >= self.emotion_update_delay:
                        try:
                            emotion_pred = asyncio.run(process_emotion_async(face_roi))
                            self.current_emotion = emotion_pred
                        except Exception as e:
                            logging.error("Error during async emotion detection: %s", e, exc_info=True)
                            self.current_emotion = "Neutral"
                        self.last_emotion_time = current_time

                    with self.data_lock:
                        self.shared_vars.ear_left = ear_left
                        self.shared_vars.ear_right = ear_right
                        self.shared_vars.mar = mar
                        self.shared_vars.lip_sync_value = lip_sync_value

            self.frame_count, fps = fps_calculation(self.frame_count, self.start_time)
            frame = self.render_overlay(frame, eye_blinked, mouth_open, lip_sync_active, self.current_emotion, fps)
            cv2.imshow("Facial Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")

def main() -> None:
    args = parse_arguments()
    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))
    config_path = args.config
    if not config.read(config_path):
        logging.error(f"Could not read config file: {config_path}")
        return

    log_level = config.get("Logging", "LOG_LEVEL", fallback="INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Facial Tracker.")

    droidcam_url = config.get("Camera", "DROIDCAM_URL", fallback="0")
    cap = cv2.VideoCapture(droidcam_url if droidcam_url != "0" else 0)
    if not cap.isOpened():
        logging.error(f"Cannot open camera: {droidcam_url}")
        return

    shared_vars = SharedVariables()
    data_lock = threading.Lock()

    websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
    websocket_thread.start()

    processor = FrameProcessor(config, shared_vars, data_lock)
    try:
        processor.process(cap)
    except Exception as e:
        logging.error("Unexpected error: %s", e, exc_info=True)
    finally:
        exit_event.set()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Cleaned up resources, exiting.")

if __name__ == "__main__":
    main()