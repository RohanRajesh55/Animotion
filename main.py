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
import json
from dataclasses import dataclass
from typing import Optional, Dict, List
from websockets.exceptions import ConnectionClosed

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logging.warning("DeepFace not installed. Emotion recognition will return 'Neutral'.")
    DEEPFACE_AVAILABLE = False

# Placeholder for test_ws module (assumed to exist)
from test_ws import send_parameter_updates, authenticate, get_token, get_current_model, get_parameter_list

@dataclass
class FaceData:
    ear_left: float
    ear_right: float
    mar: float
    eye_blinked: bool
    mouth_open: bool
    emotion: str

class SharedVariables:
    def __init__(self):
        self.fps: float = 0.0
        self.faces: List[FaceData] = []

class ConfigManager:
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            logging.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        try:
            self.config.read(config_path, encoding='utf-8')
        except UnicodeDecodeError:
            logging.error("Config file must be UTF-8 encoded")
            raise
        required_sections = ['Camera', 'WebSocket', 'Thresholds', 'Dashboard']
        for section in required_sections:
            if section not in self.config:
                logging.error(f"Missing required section in config: {section}")
                raise configparser.NoSectionError(section)

    def get_float(self, section: str, key: str, default: float) -> float:
        try:
            return self.config.getfloat(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logging.warning(f"Using default value {default} for {section}.{key}")
            return default

    def get_bool(self, section: str, key: str, default: bool) -> bool:
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logging.warning(f"Using default value {default} for {section}.{key}")
            return default

    def get_str(self, section: str, key: str, default: str) -> str:
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logging.warning(f"Using default value {default} for {section}.{key}")
            return default

    def get_int(self, section: str, key: str, default: int) -> int:
        try:
            return self.config.getint(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            logging.warning(f"Using default value {default} for {section}.{key}")
            return default

class FaceDetector:
    def __init__(self, ear_threshold: float, mar_threshold: float):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.mp_face_mesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            logging.info("FaceMesh initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize FaceMesh: {e}")
            raise

    @staticmethod
    def calculate_ear(landmarks, width: int, height: int) -> float:
        try:
            coords = [(l.x * width, l.y * height) for l in landmarks]
            v1, v2, v3, v4, v5, v6 = coords
            vert1 = np.linalg.norm(np.array(v2) - np.array(v6))
            vert2 = np.linalg.norm(np.array(v3) - np.array(v5))
            horz = np.linalg.norm(np.array(v1) - np.array(v4))
            ear = (vert1 + vert2) / (2.0 * horz) if horz != 0 else 0.0
            logging.debug(f"EAR calculated: {ear:.2f}")
            return ear
        except Exception as e:
            logging.error(f"Error calculating EAR: {e}")
            return 0.0

    @staticmethod
    def calculate_mar(landmarks, width: int, height: int) -> float:
        try:
            coords = [(l.x * width, l.y * height) for l in landmarks]
            v1, v2, v3, v4 = coords
            vert = np.linalg.norm(np.array(v3) - np.array(v4))
            horz = np.linalg.norm(np.array(v1) - np.array(v2))
            mar = vert / horz if horz != 0 else 0.0
            logging.debug(f"MAR calculated: {mar:.2f}")
            return mar
        except Exception as e:
            logging.error(f"Error calculating MAR: {e}")
            return 0.0

    def analyze_emotion(self, face_roi) -> str:
        if not DEEPFACE_AVAILABLE:
            logging.debug("DeepFace unavailable, returning 'Neutral'")
            return "Neutral"
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            emotion = result[0].get("dominant_emotion", "Neutral") if isinstance(result, list) else result.get("dominant_emotion", "Neutral")
            logging.debug(f"Emotion detected: {emotion}")
            return emotion
        except Exception as e:
            logging.error(f"Emotion analysis failed: {e}")
            return "Neutral"

    def detect(self, frame, width: int, height: int) -> tuple[List[FaceData], Optional[List]]:
        logging.debug("Starting face detection")
        if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            logging.error("Invalid frame: empty or None")
            return [], None
        try:
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
                logging.debug("Frame converted to contiguous array")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logging.debug("Frame converted to RGB")
        except cv2.error as e:
            logging.error(f"Failed to convert frame to RGB: {e}")
            return [], None
        try:
            result = self.face_mesh.process(rgb)
            logging.debug("FaceMesh processing completed")
        except Exception as e:
            logging.error(f"FaceMesh processing failed: {e}")
            return [], None
        faces = []
        if result.multi_face_landmarks:
            logging.debug(f"Detected {len(result.multi_face_landmarks)} faces")
            for lm in result.multi_face_landmarks:
                try:
                    left_eye = [lm.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                    right_eye = [lm.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                    mouth = [lm.landmark[i] for i in [61, 291, 13, 14]]
                    ear_left = self.calculate_ear(left_eye, width, height)
                    ear_right = self.calculate_ear(right_eye, width, height)
                    mar = self.calculate_mar(mouth, width, height)
                    box = np.array([[p.x * width, p.y * height] for p in lm.landmark])
                    x1, y1 = np.min(box, axis=0).astype(int)
                    x2, y2 = np.max(box, axis=0).astype(int)
                    face_roi = frame[max(0, y1):y2, max(0, x1):x2]
                    if face_roi.size > 0:
                        faces.append(FaceData(
                            ear_left=ear_left,
                            ear_right=ear_right,
                            mar=mar,
                            eye_blinked=(ear_left + ear_right) / 2 < self.ear_threshold,
                            mouth_open=mar > self.mar_threshold,
                            emotion=self.analyze_emotion(face_roi)
                        ))
                        logging.debug(f"Face processed: EAR_L={ear_left:.2f}, EAR_R={ear_right:.2f}, MAR={mar:.2f}")
                    else:
                        logging.warning("Invalid face ROI, skipping emotion analysis")
                except Exception as e:
                    logging.error(f"Error processing face landmarks: {e}")
                    continue
        else:
            logging.debug("No faces detected in frame")
        return faces, result.multi_face_landmarks

class Dashboard:
    def __init__(self, shared_vars: SharedVariables, data_lock: threading.Lock, exit_event: threading.Event):
        logging.info("Initializing Dashboard")
        self.shared_vars = shared_vars
        self.data_lock = data_lock
        self.exit_event = exit_event
        self.root = tk.Tk()
        self.root.title("Diagnostics Dashboard")
        self.label = tk.Label(self.root, text="Starting...", font=("Helvetica", 14))
        self.label.pack(padx=20, pady=20)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update(self):
        if self.exit_event.is_set():
            self.on_closing()
            return
        try:
            with self.data_lock:
                fps = self.shared_vars.fps
                faces = self.shared_vars.faces
            info = f"FPS: {fps:.2f}\nFaces Detected: {len(faces)}\n"
            if not faces:
                info += "No faces detected\n"
            else:
                for idx, face in enumerate(faces):
                    info += f"\nFace {idx+1}:\n"
                    info += f"  EAR Left : {face.ear_left:.2f}\n"
                    info += f"  EAR Right: {face.ear_right:.2f}\n"
                    info += f"  MAR      : {face.mar:.2f}\n"
                    info += f"  Blink    : {'Yes' if face.eye_blinked else 'No'}\n"
                    info += f"  Mouth Open: {'Yes' if face.mouth_open else 'No'}\n"
                    info += f"  Emotion  : {face.emotion}\n"
            self.label.config(text=info)
            self.root.after(100, self.update)
        except tk.TclError as e:
            logging.warning(f"Tkinter error: {e}")
            self.on_closing()

    def on_closing(self):
        logging.info("Dashboard window closing")
        self.exit_event.set()
        try:
            self.root.quit()
            self.root.destroy()
        except tk.TclError as e:
            logging.warning(f"Error during Tkinter cleanup: {e}")

    def run(self):
        logging.info("Starting Dashboard mainloop")
        self.update()
        try:
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Dashboard mainloop error: {e}")
            self.on_closing()

class CameraProcessor:
    def __init__(self, config: ConfigManager, shared_vars: SharedVariables, data_lock: threading.Lock, exit_event: threading.Event):
        self.config = config
        self.shared_vars = shared_vars
        self.data_lock = data_lock
        self.face_detector = FaceDetector(
            ear_threshold=config.get_float('Thresholds', 'EAR_THRESHOLD', 0.22),
            mar_threshold=config.get_float('Thresholds', 'MAR_THRESHOLD', 0.5)
        )
        self.frame_count = 0
        self.frame_times = []
        self.fps = 0.0
        self.exit_event = exit_event
        self.frame_skip = config.get_int('Camera', 'FRAME_SKIP', 1)
        logging.info("CameraProcessor initialized")

    def calculate_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        self.frame_count += 1
        self.frame_times = [t for t in self.frame_times if current_time - t <= 1.0]
        if len(self.frame_times) > 1:
            elapsed = self.frame_times[-1] - self.frame_times[0]
            fps = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0.0
            alpha = 0.1
            self.fps = alpha * fps + (1 - alpha) * self.fps if self.fps > 0 else fps
        else:
            self.fps = 0.0
        logging.debug(f"FPS calculated: {self.fps:.2f}, frames in window: {len(self.frame_times)}")
        return self.fps

    def get_latest_data(self) -> Optional[Dict]:
        with self.data_lock:
            if not self.shared_vars.faces:
                return None
            face = self.shared_vars.faces[0]
            data = {
                "MouthOpen": min(face.mar / 1.0, 1.0),
                "EyeOpenLeft": min(face.ear_left / 0.3, 1.0),
                "EyeOpenRight": min(face.ear_right / 0.3, 1.0),
            }
            return data

    def process(self, cap: cv2.VideoCapture):
        logging.info("Starting camera processing")
        frame_counter = 0
        try:
            while cap.isOpened() and not self.exit_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    logging.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
                frame = cv2.resize(frame, (480, 360))
                if frame.size == 0:
                    logging.warning("Empty frame received after resize")
                    continue
                frame_counter += 1
                if frame_counter % self.frame_skip != 0:
                    continue
                height, width = frame.shape[:2]
                faces, landmarks = self.face_detector.detect(frame, width, height)
                if faces and landmarks:
                    for face, lm in zip(faces, landmarks):
                        box = np.array([[p.x * width, p.y * height] for p in lm.landmark])
                        x1, y1 = np.min(box, axis=0).astype(int)
                        x2, y2 = np.max(box, axis=0).astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, face.emotion, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                with self.data_lock:
                    self.shared_vars.faces = faces
                    self.shared_vars.fps = self.calculate_fps()
                cv2.putText(frame, f"FPS: {self.shared_vars.fps:.2f}", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                try:
                    cv2.imshow('Facial Tracker', frame)
                except cv2.error as e:
                    logging.error(f"OpenCV imshow error: {e}")
                    break
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.exit_event.set()
                    break
        except Exception as e:
            logging.error(f"Error in camera processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            logging.info("Camera processing stopped")

def cleanup_resources(cap: Optional[cv2.VideoCapture], ws: Optional[websockets.WebSocketClientProtocol]):
    if cap is not None and cap.isOpened():
        cap.release()
        logging.info("Camera released")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    if ws is not None and not ws.closed:
        try:
            asyncio.get_event_loop().run_until_complete(ws.close())
            logging.info("WebSocket closed")
        except Exception as e:
            logging.error(f"Error closing WebSocket: {e}")
    logging.info("Resource cleanup complete")

def initialize_camera(url: str, max_retries: int = 3, retry_delay: float = 1.0) -> Optional[cv2.VideoCapture]:
    try:
        if url.isdigit():
            url = int(url)
    except ValueError:
        pass
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW]
    if isinstance(url, str):
        backends.append(cv2.CAP_FFMPEG)
    for attempt in range(max_retries):
        for backend in backends:
            backend_name = {
                cv2.CAP_ANY: "CAP_ANY",
                cv2.CAP_DSHOW: "CAP_DSHOW",
                cv2.CAP_FFMPEG: "CAP_FFMPEG"
            }.get(backend, "Unknown")
            logging.info(f"Attempting to initialize camera (try {attempt + 1}/{max_retries}, backend {backend_name}): {url}")
            try:
                cap = cv2.VideoCapture(url, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        logging.info(f"Camera initialized: {url} with backend {backend_name}")
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        logging.info(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")
                        return cap
                    else:
                        logging.warning(f"Camera opened but failed to read valid frame: {url}")
                        cap.release()
                else:
                    logging.warning(f"Failed to open camera with backend {backend_name}: {url}")
            except Exception as e:
                logging.error(f"Error initializing camera {url} with backend {backend_name}: {str(e)}")
        if attempt < max_retries - 1:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    logging.error(f"Failed to initialize camera after {max_retries} attempts: {url}")
    return None

async def connect_websocket(url: str, max_retries: int = 5, retry_delay: float = 2.0, token_file: str = "vts_token.txt"):
    token = None
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            token = f.read().strip()
        logging.info(f"Loaded token from {token_file}")
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempting WebSocket connection (try {attempt + 1}/{max_retries}): {url}")
            ws = await websockets.connect(url, ping_interval=30, ping_timeout=10)
            try:
                if not token:
                    logging.info("Requesting new token from VTube Studio")
                    token_response = await get_token(ws)
                    if not token_response or not isinstance(token_response, dict):
                        logging.error(f"Invalid token response: {token_response}")
                        raise Exception("Failed to obtain token")
                    logging.info(f"Token response: {token_response}")
                    token = token_response["data"].get("authenticationToken")
                    if not token:
                        logging.error("No authentication token in response")
                        raise Exception("No authentication token provided")
                    logging.info("New token obtained, waiting for VTube Studio approval (15 seconds)")
                    await asyncio.sleep(15)
                auth_response = await authenticate(ws, token)
                if not auth_response or not isinstance(auth_response, dict):
                    logging.error(f"Invalid auth response: {auth_response}")
                    raise Exception("Invalid authentication response")
                logging.info(f"Auth response: {auth_response}")
                if auth_response.get("messageType") != "AuthenticationResponse" or not auth_response["data"].get("authenticated"):
                    error_msg = auth_response.get("data", {}).get("message", "Unknown error")
                    logging.error(f"Authentication failed: {error_msg}")
                    if os.path.exists(token_file):
                        logging.info(f"Clearing invalid token: {token_file}")
                        os.remove(token_file)
                    token = None
                    raise Exception(f"Authentication failed: {error_msg}")
                logging.info("WebSocket connected and authenticated successfully")
                try:
                    await ws.ping()
                    logging.info("Sent post-authentication ping")
                    await asyncio.sleep(1)
                except Exception as e:
                    logging.error(f"Post-authentication ping failed: {e}")
                    raise
                return ws
            except Exception as e:
                await ws.close()
                logging.warning(f"Authentication attempt failed: {e}")
                raise
        except Exception as e:
            logging.warning(f"WebSocket connection failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                raise Exception(f"Failed to connect to WebSocket after {max_retries} attempts: {url}")

async def websocket_loop(ws, processor: CameraProcessor, model_id: str, max_retries: int = 5, retry_delay: float = 2.0):
    attempt = 0
    update_count = 0
    while not processor.exit_event.is_set() and attempt < max_retries:
        try:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                logging.info(f"Received unexpected message: {message}")
            except asyncio.TimeoutError:
                pass
            parameters = await get_parameter_list(ws, model_id)
            valid_params = {p["id"] for p in parameters}
            logging.info(f"Valid parameters: {valid_params}")
            required_params = {"MouthOpen", "EyeOpenLeft", "EyeOpenRight"}
            if not required_params.issubset(valid_params):
                logging.warning(f"Model missing required parameters: {required_params - valid_params}")
            while not processor.exit_event.is_set():
                data = processor.get_latest_data()
                if data:
                    valid_data = {k: v for k, v in data.items() if k in valid_params}
                    if valid_data:
                        update_count += 1
                        logging.debug(f"Parameter update #{update_count} at {time.strftime('%H:%M:%S')}: {valid_data}")
                        try:
                            await send_parameter_updates(ws, valid_data)
                            logging.debug(f"Successfully sent parameter update #{update_count}")
                        except Exception as e:
                            logging.error(f"Failed to send parameter update #{update_count}: {e}")
                await asyncio.sleep(1 / 30)
        except ConnectionClosed as e:
            logging.error(f"WebSocket connection closed: {e}")
            attempt += 1
            if attempt < max_retries:
                logging.info(f"Retrying WebSocket connection (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                try:
                    ws = await connect_websocket(ws.uri, max_retries, retry_delay)
                    logging.info("WebSocket reconnected successfully")
                except Exception as reconnect_error:
                    logging.error(f"Failed to reconnect WebSocket: {reconnect_error}")
                    processor.exit_event.set()
                    break
            else:
                logging.error("Max retries reached, shutting down")
                processor.exit_event.set()
        except Exception as e:
            logging.error(f"Error in websocket_loop: {e}")
            processor.exit_event.set()
            break
        finally:
            if not ws.closed:
                try:
                    await ws.close()
                    logging.info("WebSocket closed in websocket_loop")
                except Exception as e:
                    logging.error(f"Error closing WebSocket in websocket_loop: {e}")

async def async_main(exit_event: threading.Event, config_path: str = "config.ini"):
    parser = argparse.ArgumentParser(description="Real-Time Facial Tracker")
    parser.add_argument('--config', type=str, default=config_path, help='Path to configuration file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        config = ConfigManager(args.config)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        exit_event.set()
        return
    cap = initialize_camera(config.get_str('Camera', 'DROIDCAM_URL', '0'), max_retries=3, retry_delay=1.0)
    if not cap:
        logging.error("Camera initialization failed. Exiting.")
        exit_event.set()
        return
    shared_vars = SharedVariables()
    data_lock = threading.Lock()
    processor = CameraProcessor(config, shared_vars, data_lock, exit_event)
    logging.info("CameraProcessor created, checking methods")
    if not hasattr(processor, 'process'):
        logging.error("CameraProcessor missing 'process' method")
        exit_event.set()
        return
    ws = None
    try:
        ws_url = config.get_str('WebSocket', 'URL', 'ws://localhost:8001')
        logging.info(f"Attempting to connect to WebSocket at {ws_url}")
        ws = await connect_websocket(ws_url, max_retries=5, retry_delay=2.0)
        model_id = await get_current_model(ws)
        if not model_id:
            logging.error("No model loaded in VTube Studio. Please load a model and retry.")
            exit_event.set()
            return
        logging.info(f"Current model ID: {model_id}")
        camera_thread = threading.Thread(target=processor.process, args=(cap,), daemon=True)
        camera_thread.start()
        await websocket_loop(ws, processor, model_id, max_retries=5, retry_delay=2.0)
    except ConnectionClosed as e:
        logging.error(f"WebSocket connection closed: {e}")
        exit_event.set()
    except Exception as e:
        logging.error(f"Error in async main loop: {e}")
        exit_event.set()
    finally:
        cleanup_resources(cap, ws)

def main():
    exit_event = threading.Event()
    def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}, shutting down")
        exit_event.set()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        config = ConfigManager('config.ini')
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return
    if config.get_bool('Dashboard', 'ENABLE_DASHBOARD', True):
        shared_vars = SharedVariables()
        data_lock = threading.Lock()
        # Check if Dashboard class is defined
        if not globals().get('Dashboard'):
            logging.error("Dashboard class is not defined")
            return
        try:
            dashboard = Dashboard(shared_vars, data_lock, exit_event)
            logging.info("Dashboard instantiated successfully")
        except Exception as e:
            logging.error(f"Failed to instantiate Dashboard: {e}")
            return
        asyncio_thread = threading.Thread(
            target=lambda: asyncio.run(async_main(exit_event)),
            daemon=True
        )
        asyncio_thread.start()
        dashboard.run()
        asyncio_thread.join()
    else:
        asyncio.run(async_main(exit_event))

if __name__ == "__main__":
    main()