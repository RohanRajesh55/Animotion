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
import asyncio
import os
import websockets
import uuid

from test_ws import send_parameter_updates, authenticate, get_token
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

# Try to load DeepFace for emotion detection
try:
    from deepface import DeepFace
    deepface_available = True
except ImportError:
    logging.warning("DeepFace is not installed. Emotion recognition will return 'Neutral'.")
    deepface_available = False

exit_event = threading.Event()

def signal_handler(sig, frame):
    logging.info("Signal received. Exiting gracefully...")
    exit_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-Time Facial Tracker for Avatar Animation")
    parser.add_argument('--config', type=str, default='config.ini', help='Path to configuration file')
    return parser.parse_args()

def start_dashboard(shared_vars, data_lock):
    root = tk.Tk()
    root.title("Diagnostics Dashboard")
    label = tk.Label(root, text="Starting...", font=("Helvetica", 14))
    label.pack(padx=20, pady=20)

    def update():
        with data_lock:
            fps = shared_vars.fps if shared_vars.fps else 0.0
            faces = shared_vars.faces
        info = f"FPS: {fps:.2f}\nFaces Detected: {len(faces)}\n"
        for idx, face in enumerate(faces):
            info += f"\nFace {idx+1}:\n"
            info += f"  EAR Left : {face.get('ear_left', 'N/A'):.2f}\n"
            info += f"  EAR Right: {face.get('ear_right', 'N/A'):.2f}\n"
            info += f"  MAR      : {face.get('mar', 'N/A'):.2f}\n"
            info += f"  Blink    : {'Yes' if face.get('eye_blinked', False) else 'No'}\n"
            info += f"  Mouth Open: {'Yes' if face.get('mouth_open', False) else 'No'}\n"
            info += f"  Emotion  : {face.get('emotion', 'N/A')}\n"
        label.config(text=info)
        root.after(1000, update)

    update()
    root.mainloop()

class FrameProcessor:
    def __init__(self, config, shared_vars, data_lock):
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock

        self.ear_threshold = config.getfloat('Thresholds', 'EAR_THRESHOLD', fallback=0.22)
        self.mar_threshold = config.getfloat('Thresholds', 'MAR_THRESHOLD', fallback=0.5)

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frame_count = 0
        self.start_time = time.time()

    def analyze_emotion(self, face_roi):
        if deepface_available:
            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                if isinstance(result, list): result = result[0]
                return result.get("dominant_emotion", "Neutral")
            except Exception as e:
                logging.error(f"Emotion analysis error: {e}")
        return "Neutral"

    def detect_eye_blink(self, landmarks, width, height):
        left = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
        right = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        ear_l = calculate_ear(left, width, height)
        ear_r = calculate_ear(right, width, height)
        return ear_l, ear_r, (ear_l + ear_r) / 2 < self.ear_threshold

    def get_latest_data(self):
        with self.data_lock:
            if not self.shared_vars.faces:
                return None
            face = self.shared_vars.faces[0]
            return {
                "MouthOpen": min(face["mar"] / 1.0, 1.0),
                "EyeOpenLeft": min(face["ear_left"] / 0.3, 1.0),
                "EyeOpenRight": min(face["ear_right"] / 0.3, 1.0),
            }

    def process(self, cap):
        while cap.isOpened() and not exit_event.is_set():
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)
            h, w = frame.shape[:2]
            faces = []

            if result.multi_face_landmarks:
                for lm in result.multi_face_landmarks:
                    ear_l, ear_r, blink = self.detect_eye_blink(lm, w, h)
                    mar = calculate_mar([lm.landmark[i] for i in [61, 291, 13, 14]], w, h)
                    box = np.array([[p.x * w, p.y * h] for p in lm.landmark])
                    x1, y1 = np.min(box, axis=0).astype(int)
                    x2, y2 = np.max(box, axis=0).astype(int)
                    face_img = frame[y1:y2, x1:x2]
                    emotion = self.analyze_emotion(face_img)
                    faces.append({
                        "ear_left": ear_l, "ear_right": ear_r, "mar": mar,
                        "eye_blinked": blink, "mouth_open": mar > self.mar_threshold,
                        "emotion": emotion
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, emotion, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            with self.data_lock:
                self.shared_vars.faces = faces
            self.frame_count, fps = fps_calculation(self.frame_count, self.start_time)
            with self.data_lock:
                self.shared_vars.fps = fps

            cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Facial Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()

async def tracking_loop(ws, processor):
    while not exit_event.is_set():
        data = processor.get_latest_data()
        if data:
            await send_parameter_updates(ws, data)
        await asyncio.sleep(1 / 30)

def main():
    args = parse_arguments()
    config = configparser.ConfigParser()
    config.read(args.config)

    logging.basicConfig(level=logging.INFO)
    droidcam_url = config.get('Camera', 'DROIDCAM_URL', fallback='0')
    cap = cv2.VideoCapture(droidcam_url if droidcam_url != '0' else 0)

    if not cap.isOpened():
        logging.error(f"Cannot open camera: {droidcam_url}")
        return

    shared_vars = SharedVariables()
    shared_vars.faces = []
    shared_vars.fps = 0.0
    data_lock = threading.Lock()
    processor = FrameProcessor(config, shared_vars, data_lock)

    if config.getboolean('Dashboard', 'ENABLE_DASHBOARD', fallback=True):
        threading.Thread(target=start_dashboard, args=(shared_vars, data_lock), daemon=True).start()

    async def run():
        try:
            async with websockets.connect("ws://localhost:8001") as ws:
                if os.path.exists("vts_token.txt"):
                    with open("vts_token.txt") as f:
                        token = f.read().strip()
                else:
                    token = await get_token(ws)
                await authenticate(ws, token)

                # Start camera processing
                tracking_thread = threading.Thread(target=processor.process, args=(cap,), daemon=True)
                tracking_thread.start()

                # Send tracking data to VTube Studio
                await tracking_loop(ws, processor)

        except Exception as e:
            logging.error(f"WebSocket or processing error: {e}")
        finally:
            exit_event.set()
            cap.release()
            cv2.destroyAllWindows()

    asyncio.run(run())

if __name__ == "__main__":
    main()
