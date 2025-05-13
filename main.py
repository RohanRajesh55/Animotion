#!/usr/bin/env python3
"""
Facial Tracker - Production Entry Point

This module implements a real‑time facial animation tracker that captures a camera feed,
processes facial landmarks via MediaPipe, calculates eye/mouth metrics, and analyzes emotion using DeepFace.
It also streams filtered data over a WebSocket connection on a separate thread.
"""

import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import configparser
import logging
import signal
import argparse

from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.lip_sync import calculate_lip_sync_value
from websocket_client import start_websocket
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables
from filter.kalman_filter import KalmanFilter
from deepface import DeepFace

# Global event used to gracefully terminate the application.
exit_event = threading.Event()


def signal_handler(sig, frame):
    logging.info("Signal received. Exiting gracefully...")
    exit_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Live Face-to-Animation Tracker")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to configuration file")
    return parser.parse_args()


class FrameProcessor:
    def __init__(self, config, shared_vars: SharedVariables, data_lock: threading.Lock) -> None:
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        # Retrieve configuration thresholds and parameters.
        self.ear_threshold = config.getfloat("Thresholds", "EAR_THRESHOLD", fallback=0.22)
        self.mar_threshold = config.getfloat("Thresholds", "MAR_THRESHOLD", fallback=0.5)
        self.emotion_analysis_interval = config.getint("Advanced", "EMOTION_ANALYSIS_INTERVAL", fallback=15)
        self.run_emotion = config.getboolean("Advanced", "RUN_EMOTION_ANALYSIS", fallback=True)

        # Initialize MediaPipe Face Mesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Kalman filters for smoothing various measurements.
        self.ear_kalman = KalmanFilter()
        self.mar_kalman = KalmanFilter()
        self.lip_sync_kalman = KalmanFilter()

        # History buffers for adaptive thresholding and smoothing.
        self.ear_history = []
        self.lip_history = []
        self.emotion_history = []

        self.frame_count = 0
        self.start_time = time.time()

    def smooth_emotion(self, emotion_text: str) -> str:
        if len(self.emotion_history) >= 10:
            self.emotion_history.pop(0)
        self.emotion_history.append(emotion_text)
        # Return the emotion that appears most frequently in the history buffer.
        return max(set(self.emotion_history), key=self.emotion_history.count)

    def process_landmarks(self, face_landmarks, width: int, height: int):
        # Define indices for each facial component.
        left_indices = [33, 160, 158, 133, 153, 144]
        right_indices = [362, 385, 387, 263, 373, 380]
        mar_indices = [61, 291, 13, 14]
        lip_sync_indices = [13, 14, 61, 291]

        # Calculate raw metrics.
        raw_ear_left = calculate_ear([face_landmarks.landmark[i] for i in left_indices], width, height)
        raw_ear_right = calculate_ear([face_landmarks.landmark[i] for i in right_indices], width, height)
        raw_mar = calculate_mar([face_landmarks.landmark[i] for i in mar_indices], width, height)
        raw_lip_sync = calculate_lip_sync_value([face_landmarks.landmark[i] for i in lip_sync_indices], width, height)

        # Apply Kalman filters for smoothing.
        ear_left = self.ear_kalman.update(raw_ear_left)
        ear_right = self.ear_kalman.update(raw_ear_right)
        mar = self.mar_kalman.update(raw_mar)
        lip_sync_value = self.lip_sync_kalman.update(raw_lip_sync)

        # Update adaptive threshold using a rolling history.
        avg_ear = (ear_left + ear_right) / 2
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 50:
            self.ear_history.pop(0)
        adaptive_ear_threshold = max(min(self.ear_history) * 0.85, 0.18)
        eye_blinked = raw_ear_left < adaptive_ear_threshold and raw_ear_right < adaptive_ear_threshold

        # Smooth lip sync measurement.
        self.lip_history.append(lip_sync_value)
        if len(self.lip_history) > 30:
            self.lip_history.pop(0)
        smoothed_lip_sync = sum(self.lip_history[-10:]) / min(len(self.lip_history), 10)
        lip_sync_active = smoothed_lip_sync > 0.12

        mouth_open = mar > self.mar_threshold

        return eye_blinked, mouth_open, lip_sync_active, ear_left, ear_right, mar, lip_sync_value

    def analyze_emotion(self, face_roi: np.ndarray) -> str:
        emotion_text = "None"
        try:
            if face_roi.size != 0 and self.run_emotion:
                results = DeepFace.analyze(
                    face_roi,
                    actions=["emotion"],
                    enforce_detection=True,
                    silent=True
                )
                # Support both dict and list response formats.
                if isinstance(results, dict) and "dominant_emotion" in results:
                    emotion_text = results["dominant_emotion"]
                elif isinstance(results, list) and results and "dominant_emotion" in results[0]:
                    emotion_text = results[0]["dominant_emotion"]
                else:
                    emotion_text = "No Emotion"
            elif face_roi.size == 0:
                emotion_text = "No Face"
        except Exception as e:
            logging.error(f"Emotion detection error: {e}", exc_info=True)
            emotion_text = "Error"
        return emotion_text

    def render_overlay(self, frame, eye_status: bool, mouth_status: bool, lip_status: bool, emotion_text: str, fps: float):
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

            # Resize frame for a consistent processing resolution.
            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]

            # (Optional) Prepare camera matrix if needed for further processing.
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            # Reset measurement defaults for the current frame.
            eye_blinked = False
            mouth_open = False
            lip_sync_active = False
            emotion_text = "None"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    (eye_blinked, mouth_open, lip_sync_active,
                     ear_left, ear_right, mar, lip_sync_value) = self.process_landmarks(face_landmarks, width, height)

                    # Extract facial ROI for emotion analysis.
                    landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                    x1, y1 = np.min(landmarks_array, axis=0).astype(int)
                    x2, y2 = np.max(landmarks_array, axis=0).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    face_roi = frame[y1:y2, x1:x2]

                    if self.frame_count % self.emotion_analysis_interval == 0:
                        detected_emotion = self.analyze_emotion(face_roi)
                        emotion_text = self.smooth_emotion(detected_emotion)

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
    args = parse_arguments()

    # Load configuration with error-checking.
    config = configparser.ConfigParser()
    files_read = config.read(args.config)
    if not files_read:
        logging.error(f"Configuration file {args.config} not found or unreadable.")
        return

    # Configure logging.
    log_level = config.get("Logging", "LOG_LEVEL", fallback="INFO").upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Facial Tracker.")

    # Open the camera resource (supporting both device and URL streams).
    droidcam_url = config.get("Camera", "DROIDCAM_URL", fallback="0")
    cap = cv2.VideoCapture(droidcam_url if droidcam_url != "0" else 0)
    if not cap.isOpened():
        logging.error(f"Cannot open camera '{droidcam_url}'")
        return

    # Create shared variables and a thread-safe lock.
    shared_vars = SharedVariables()
    data_lock = threading.Lock()

    # Start the WebSocket client on a daemon thread.
    websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
    websocket_thread.start()

    # Initiate frame processing.
    processor = FrameProcessor(config, shared_vars, data_lock)
    try:
        processor.process(cap)
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}", exc_info=True)
    finally:
        exit_event.set()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources have been cleaned up, exiting.")


if __name__ == "__main__":
    main()