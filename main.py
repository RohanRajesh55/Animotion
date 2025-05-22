import cv2
import time
import logging
from tkinter import Tk, Label, StringVar, Frame
from PIL import Image, ImageTk

from detectors.facial_landmarks_processor import FacialLandmarksProcessor
from emotion_recognition import recognize_emotion
from utils.shared_variables import SharedVariables
from utils.config_manager import ConfigManager

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration from config.ini.
config = ConfigManager("config.ini")

# Retrieve camera and threshold settings from the configuration.
video_source = config.get_int("Camera", "DROIDCAM_URL", fallback=0)

# Thresholds for detection.
EAR_THRESHOLD = config.get_float("Thresholds", "EAR_THRESHOLD", fallback=0.190)
MAR_THRESHOLD = config.get_float("Thresholds", "MAR_THRESHOLD", fallback=0.083)
EBR_THRESHOLD = config.get_float("Thresholds", "EBR_THRESHOLD", fallback=1.5)
EMOTION_THRESHOLD = config.get_float("Thresholds", "EMOTION_THRESHOLD", fallback=0.4)

# Advanced settings.
EMOTION_ANALYSIS_INTERVAL = config.get_int("Advanced", "EMOTION_ANALYSIS_INTERVAL", fallback=15)

# GPU availability & processing settings.
has_gpu = False
try:
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        has_gpu = True
        logger.info("GPU detected – utilizing GPU acceleration if available.")
    else:
        logger.info("No GPU detected – running on CPU.")
except Exception as e:
    logger.info("Error checking GPU: %s", e)

if has_gpu:
    VIDEO_UPDATE_DELAY = 10        # Faster update rate
    SKIP_PROCESSING_FRAMES = 1       # Process every frame
else:
    VIDEO_UPDATE_DELAY = 15        # Try to achieve roughly 30 FPS on CPU
    SKIP_PROCESSING_FRAMES = 4     # Skip a few frames for heavy processing

class AnimotionInterface:
    def __init__(self, video_source: int = video_source) -> None:
        """
        Initialize the Animotion interface: set up video capture, processing,
        and build the Tkinter dashboard.
        """
        self.video_source = video_source
        self.shared_vars = SharedVariables()
        self.processor = FacialLandmarksProcessor()

        # Initialize video capture.
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            logger.error("Cannot open video source: %s", self.video_source)
            raise Exception("Video capture initialization failed.")
        # Set a lower resolution to help with performance.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize Tkinter dashboard.
        self.root = Tk()
        self.root.title("Animotion - Facial Tracking Interface")
        self.root.bind("<Key>", self.on_key_press)

        # Video display area.
        self.video_label = Label(self.root)
        self.video_label.pack()

        # Status display area.
        status_frame = Frame(self.root)
        status_frame.pack(pady=5)

        self.status_var = StringVar()
        self.status_var.set("Running")
        self.status_label = Label(status_frame, textvariable=self.status_var, font=("Arial", 12))
        self.status_label.pack()

        self.detection_status_var = StringVar()
        self.detection_status_var.set("Detections: N/A")
        self.detection_status_label = Label(status_frame, textvariable=self.detection_status_var, font=("Arial", 12))
        self.detection_status_label.pack()

        self.fps_var = StringVar()
        self.fps_var.set("FPS: N/A")
        self.fps_label = Label(status_frame, textvariable=self.fps_var, font=("Arial", 12))
        self.fps_label.pack()

        # Internal state for managing processing.
        self.running = False
        self.failed_frame_count = 0
        self.last_emotion_time = 0.0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.skip_counter = 0

    def on_key_press(self, event):
        """Stop processing and close the window when the 'q' key is pressed."""
        if event.char.lower() == 'q':
            self.stop()
            self.root.destroy()

    def start(self) -> None:
        """Begin video processing."""
        if self.running:
            return
        self.running = True
        self.last_emotion_time = time.time()
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.skip_counter = 0
        self.update_video()  # Begin the update loop.
        logger.info("Animotion interface started.")

    def stop(self) -> None:
        """Stop the video processing loop."""
        self.running = False
        self.status_var.set("Stopped")
        logger.info("Animotion interface stopped.")

    def reinitialize_video_capture(self) -> None:
        """Reinitialize the video capture if too many frames fail."""
        try:
            self.cap.release()
        except Exception as e:
            logger.error("Error releasing video capture: %s", e)
        logger.info("Reinitializing video capture on source %s...", self.video_source)
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            logger.error("Reinitialization failed; unable to open video source %s.", self.video_source)
        else:
            logger.info("Video capture reinitialized successfully.")

    def update_video(self) -> None:
        """
        Capture a video frame, run detection and emotion recognition (on selected frames),
        update shared metrics, update UI, and schedule the next update.
        """
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.failed_frame_count += 1
            logger.warning("Failed to capture frame. Consecutive failures: %s", self.failed_frame_count)
            if self.failed_frame_count >= config.get_int("Thresholds", "FAILED_FRAME_THRESHOLD", fallback=10):
                logger.warning("Reinitializing video capture after too many failures.")
                self.reinitialize_video_capture()
                self.failed_frame_count = 0
            self.root.after(VIDEO_UPDATE_DELAY, self.update_video)
            return
        else:
            self.failed_frame_count = 0

        self.frame_count += 1
        self.skip_counter += 1
        current_time = time.time()

        # Run heavy processing only on selected frames.
        if self.skip_counter % SKIP_PROCESSING_FRAMES == 0:
            metrics = self.processor.process_frame(frame)
            if metrics:
                # Update shared variables from computed metrics.
                self.shared_vars.ear_left = metrics.get("ear_left")
                self.shared_vars.ear_right = metrics.get("ear_right")
                self.shared_vars.mar = metrics.get("mar")
                self.shared_vars.ebr_left = metrics.get("ebr_left")
                self.shared_vars.ebr_right = metrics.get("ebr_right")
                self.shared_vars.lip_sync_value = metrics.get("lip_sync_value")
                head_pose = metrics.get("head_pose", {})
                if head_pose.get("rotation_vector") is not None:
                    rotation = head_pose["rotation_vector"]
                    if isinstance(rotation, (list, tuple)) and len(rotation) >= 3:
                        self.shared_vars.yaw = rotation[0]
                        self.shared_vars.pitch = rotation[1]
                        self.shared_vars.roll = rotation[2]
            # Run emotion recognition based on interval.
            if current_time - self.last_emotion_time > EMOTION_ANALYSIS_INTERVAL:
                try:
                    emotion = recognize_emotion(frame)
                except Exception as e:
                    logger.error("Emotion recognition failed: %s", e)
                    emotion = "unknown"
                self.shared_vars.emotion = emotion
                self.last_emotion_time = current_time

        # Build a status string to display detection outcomes.
        detection_text = ""
        if self.shared_vars.ear_left is not None and self.shared_vars.ear_right is not None:
            avg_ear = (self.shared_vars.ear_left + self.shared_vars.ear_right) / 2.0
            blink_det = "Yes" if avg_ear < EAR_THRESHOLD else "No"
            detection_text += f"Blink: {blink_det}\n"
        else:
            detection_text += "Blink: N/A\n"

        if self.shared_vars.mar is not None:
            mouth_status = "Yes" if self.shared_vars.mar > MAR_THRESHOLD else "No"
            detection_text += f"Mouth Open: {mouth_status}\n"
        else:
            detection_text += "Mouth Open: N/A\n"

        emotion_disp = self.shared_vars.emotion if self.shared_vars.emotion is not None else "N/A"
        detection_text += f"Emotion: {emotion_disp}"
        self.detection_status_var.set(detection_text)

        # Update FPS display every second.
        elapsed_time = current_time - self.last_fps_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.fps_var.set(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_fps_time = current_time

        # Convert frame color and update UI.
        try:
            cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # Keep a reference.
            self.video_label.configure(image=imgtk)
        except Exception as e:
            logger.error("Error converting frame for display: %s", e)

        self.root.after(VIDEO_UPDATE_DELAY, self.update_video)

    def run(self) -> None:
        """Start the interface and begin the Tkinter event loop."""
        self.start()
        self.root.mainloop()

def main() -> None:
    interface = AnimotionInterface(video_source=video_source)
    interface.run()

if __name__ == '__main__':
    main()