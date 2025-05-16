import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import configparser
import logging
import signal
import argparse
import tkinter as tk
import collections

from emotion_recognition import analyze_emotion
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

# Global exit event for graceful termination.
exit_event = threading.Event()

def signal_handler(sig, frame):
    logging.info("Signal received. Exiting gracefully...")
    exit_event.set()

# Register termination signals.
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Facial Tracker for Avatar Animation")
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    return parser.parse_args()

def start_dashboard(shared_vars, data_lock):
    """
    Starts a Tkinter diagnostics dashboard that updates every second with the current FPS 
    and face detection details.
    """
    root = tk.Tk()
    root.title("Diagnostics Dashboard")
    label = tk.Label(root, text="Starting...", font=("Helvetica", 14))
    label.pack(padx=20, pady=20)

    def update():
        with data_lock:
            fps = shared_vars.fps if shared_vars.fps is not None else 0.0
            faces = shared_vars.faces
        info = f"FPS: {fps:.2f}\nFaces Detected: {len(faces)}\n"
        for idx, face in enumerate(faces):
            ear_left = face.get("ear_left", "N/A")
            ear_right = face.get("ear_right", "N/A")
            mar = face.get("mar", "N/A")
            blink = face.get("eye_blinked", False)
            mouth_open = face.get("mouth_open", False)
            emotion = face.get("emotion", "N/A")
            info += f"\nFace {idx+1}:\n"
            info += f"  EAR Left : {ear_left if isinstance(ear_left, str) else f'{ear_left:.2f}'}\n"
            info += f"  EAR Right: {ear_right if isinstance(ear_right, str) else f'{ear_right:.2f}'}\n"
            info += f"  Blink    : {'Yes' if blink else 'No'}\n"
            info += f"  MAR      : {mar if isinstance(mar, str) else f'{mar:.2f}'}\n"
            info += f"  Mouth Open: {'Yes' if mouth_open else 'No'}\n"
            info += f"  Emotion  : {emotion}\n"
        label.config(text=info)
        root.after(1000, update)

    update()
    root.mainloop()

def get_mode(emotions_list):
    """
    Helper function to calculate the mode (most common emotion) from a list.
    """
    if not emotions_list:
        return "Neutral"
    data = collections.Counter(emotions_list)
    return data.most_common(1)[0][0]

class FrameProcessor:
    """
    Processes video frames to detect facial landmarks using MediaPipe,
    calculates various metrics (EAR, MAR), detects eye blink and mouth open status,
    and integrates emotion recognition with temporal smoothing.
    """
    def __init__(self, config, shared_vars, data_lock):
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        # Read thresholds and parameters from config.
        self.ear_threshold = config.getfloat('Thresholds', 'EAR_THRESHOLD', fallback=0.22)
        self.mar_threshold = config.getfloat('Thresholds', 'MAR_THRESHOLD', fallback=0.5)
        # Number of frames to skip between emotion analyses.
        self.emotion_analysis_interval = config.getint('Advanced', 'EMOTION_ANALYSIS_INTERVAL', fallback=15)
        self.use_emotion_recognition = config.getboolean('Emotion', 'USE_EMOTION_RECOGNITION', fallback=True)
        self.run_emotion = config.getboolean('Advanced', 'RUN_EMOTION_ANALYSIS', fallback=True)
        # Buffer size for temporal smoothing of emotion results.
        self.emotion_buffer_size = config.getint('Advanced', 'EMOTION_BUFFER_SIZE', fallback=5)

        # Dictionary to store recent emotion results per face index.
        self.emotion_buffers = {}

        # Configure MediaPipe FaceMesh.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frame_count = 0
        self.start_time = time.time()

    def detect_eye_blink(self, face_landmarks, width, height, threshold):
        """Detects eye blink via the average eye aspect ratio (EAR) for both eyes."""
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
        right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
        ear_left = calculate_ear(left_eye_landmarks, width, height)
        ear_right = calculate_ear(right_eye_landmarks, width, height)
        avg_ear = (ear_left + ear_right) / 2.0
        return avg_ear < threshold

    def process(self, cap):
        """
        Main loop. Processes each video frame, detects faces and their landmarks,
        computes metrics, and performs emotion recognition (with temporal smoothing
        and reduced frequency) before displaying the annotated frame.
        """
        while cap.isOpened() and not exit_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to grab frame. Retrying...")
                continue

            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            face_data = []

            if results.multi_face_landmarks:
                for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                    # Calculate EAR and MAR from the landmarks.
                    ear_left = calculate_ear(
                        [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]],
                        width, height
                    )
                    ear_right = calculate_ear(
                        [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]],
                        width, height
                    )
                    mar = calculate_mar(
                        [face_landmarks.landmark[i] for i in [61, 291, 13, 14]],
                        width, height
                    )
                    eye_blinked = self.detect_eye_blink(face_landmarks, width, height, self.ear_threshold)
                    mouth_open = mar > self.mar_threshold

                    # Compute bounding box for the face.
                    landmarks_array = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark])
                    x1, y1 = np.min(landmarks_array, axis=0).astype(int)
                    x2, y2 = np.max(landmarks_array, axis=0).astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    face_roi = frame[y1:y2, x1:x2]

                    # Use temporal smoothing: run emotion detection every N frames.
                    if self.frame_count % self.emotion_analysis_interval == 0:
                        detected_emotion = analyze_emotion(
                            face_roi,
                            use_emotion_recognition=self.use_emotion_recognition
                        )
                        # Initialize/update buffer for this face index.
                        if idx not in self.emotion_buffers:
                            self.emotion_buffers[idx] = []
                        self.emotion_buffers[idx].append(detected_emotion)
                        if len(self.emotion_buffers[idx]) > self.emotion_buffer_size:
                            self.emotion_buffers[idx].pop(0)
                    
                    # Retrieve the smoothed emotion (mode over the buffer) or fallback to Neutral.
                    current_buffer = self.emotion_buffers.get(idx, [])
                    emotion_text = get_mode(current_buffer) if current_buffer else "Neutral"

                    face_data.append({
                        "ear_left": ear_left,
                        "ear_right": ear_right,
                        "mar": mar,
                        "eye_blinked": eye_blinked,
                        "mouth_open": mouth_open,
                        "emotion": emotion_text,
                        "bounding_box": (x1, y1, x2, y2)
                    })

                    # Annotate the frame with bounding box and emotion label.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, emotion_text, (x1, max(y1 - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            with self.data_lock:
                self.shared_vars.faces = face_data

            self.frame_count, fps = fps_calculation(self.frame_count, self.start_time)
            with self.data_lock:
                self.shared_vars.fps = fps

            cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Facial Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        logging.info("Program terminated.")

def main():
    args = parse_arguments()
    config = configparser.ConfigParser()
    config.read(args.config)

    log_level = config.get('Logging', 'LOG_LEVEL', fallback='INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Facial Tracker.")

    camera_url = config.get('Camera', 'DROIDCAM_URL', fallback='0')
    cap = cv2.VideoCapture(camera_url if camera_url != '0' else 0)
    if not cap.isOpened():
        logging.error(f"Cannot open camera '{camera_url}'")
        return

    shared_vars = SharedVariables()
    shared_vars.faces = []
    shared_vars.fps = 0.0
    data_lock = threading.Lock()

    # Start the WebSocket thread if enabled.
    if config.getboolean('WebSocket', 'ENABLE_WEBSOCKET', fallback=False):
        try:
            from websocket_client import start_websocket
            websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
            websocket_thread.start()
        except ImportError as e:
            logging.error(f"WebSocket module import failed: {e}")
    else:
        logging.info("WebSocket integration is disabled via config.")

    # Start diagnostics dashboard if enabled.
    if config.getboolean('Dashboard', 'ENABLE_DASHBOARD', fallback=True):
        dashboard_thread = threading.Thread(target=start_dashboard, args=(shared_vars, data_lock), daemon=True)
        dashboard_thread.start()
    else:
        logging.info("Diagnostics dashboard is disabled via config.")

    processor = FrameProcessor(config, shared_vars, data_lock)
    try:
        processor.process(cap)
    except Exception as e:
        logging.exception(f"Unexpected error during processing: {e}")
    finally:
        exit_event.set()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()