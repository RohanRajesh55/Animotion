import cv2
import mediapipe as mp
import time
import logging
import argparse
from detectors.eye_detector import calculate_ear
from detectors.mouth_detector import calculate_mar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def calibrate_camera(num_frames: int = 100, camera_index: int = 0) -> None:
    """
    Calibrates the camera by computing average Eye Aspect Ratio (EAR) and 
    Mouth Aspect Ratio (MAR) over a specified number of frames where the 
    subject maintains a neutral face pose.

    Args:
        num_frames (int): Number of frames to use for calibration.
        camera_index (int): The index of the camera (default is 0).
    
    Logs the average EAR and MAR, along with suggested thresholds.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    ear_values = []
    mar_values = []
    count = 0

    logging.info("Starting calibration: Please maintain a neutral face pose...")
    try:
        while count < num_frames:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame")
                break

            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                ear_left = calculate_ear(
                    [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]],
                    width,
                    height
                )
                ear_right = calculate_ear(
                    [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]],
                    width,
                    height
                )
                mar = calculate_mar(
                    [face_landmarks.landmark[i] for i in [61, 291, 13, 14]],
                    width,
                    height
                )
                avg_ear = (ear_left + ear_right) / 2.0

                ear_values.append(avg_ear)
                mar_values.append(mar)
                count += 1

                cv2.putText(
                    frame,
                    f'Calibrating: {count}/{num_frames}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Calibration terminated by user.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if ear_values and mar_values:
        avg_ear_final = sum(ear_values) / len(ear_values)
        avg_mar_final = sum(mar_values) / len(mar_values)
        # Apply margin factors to suggest thresholds.
        suggested_ear_threshold = avg_ear_final * 0.9
        suggested_mar_threshold = avg_mar_final * 1.1
        logging.info("Calibration complete!")
        logging.info(f"Average EAR: {avg_ear_final:.3f} -> Suggested EAR threshold: {suggested_ear_threshold:.3f}")
        logging.info(f"Average MAR: {avg_mar_final:.3f} -> Suggested MAR threshold: {suggested_mar_threshold:.3f}")
    else:
        logging.error("Calibration failed: No face detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera Calibration Tool")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames for calibration")
    parser.add_argument("--camera_index", type=int, default=0, help="Index of the camera (default: 0)")
    args = parser.parse_args()

    calibrate_camera(num_frames=args.num_frames, camera_index=args.camera_index)