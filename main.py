import cv2
import numpy as np
import mediapipe as mp
import torch
import time
import threading
import configparser
import logging
from torchvision import transforms
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar
from detectors.lip_sync import calculate_lip_sync_value
from detectors.head_pose_estimator import get_head_pose
from backend.websocket_client import start_websocket
from utils.calculations import fps_calculation
from utils.shared_variables import SharedVariables

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load trained AI models
expression_model = torch.load("models/facial_expression_model.pth", map_location=torch.device('cpu'))
gesture_model = torch.load("models/gesture_lstm.pth", map_location=torch.device('cpu'))
expression_model.eval()
gesture_model.eval()

# Define preprocessing for expressions
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open camera")
    exit(1)

frame_count = 0
start_time = time.time()
data_lock = threading.Lock()
shared_vars = SharedVariables()

# Start WebSocket thread
websocket_thread = threading.Thread(target=start_websocket, args=(shared_vars, data_lock), daemon=True)
websocket_thread.start()

def process_frames():
    global frame_count, start_time
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to grab frame. Retrying...")
            continue
        
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear = calculate_ear([face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]], 640, 480)
                mar = calculate_mar([face_landmarks.landmark[i] for i in [61, 291, 13, 14]], 640, 480)
                lip_sync_value = calculate_lip_sync_value([face_landmarks.landmark[i] for i in [13, 14, 61, 291]], 640, 480)
                head_pose = get_head_pose(face_landmarks, 640, 480, np.eye(3), np.zeros((4, 1)))
                
                # Expression Recognition
                face_tensor = transform(frame).unsqueeze(0)
                with torch.no_grad():
                    expression_prediction = expression_model(face_tensor)
                    expression_label = torch.argmax(expression_prediction, dim=1).item()

                # Gesture Recognition
                gesture_tensor = torch.tensor([ear, mar, lip_sync_value], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    gesture_prediction = gesture_model(gesture_tensor)
                    gesture_label = torch.argmax(gesture_prediction, dim=1).item()

                # Overlay Results on Frame
                cv2.putText(frame, f"Expression: {expression_label}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {gesture_label}", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Head Pose: {head_pose}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
        frame_count, fps = fps_calculation(frame_count, start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Animotion Facial Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Program terminated.")

if __name__ == "__main__":
    process_frames()

    # Wait for WebSocket thread to finish
    websocket_thread.join()
    logger.info("WebSocket thread terminated.")
    shared_vars.websocket_connected.set()
    shared_vars.websocket_thread.join()
    logger.info("WebSocket server disconnected.")
    logger.info("Program terminated.")
    exit(0)
    