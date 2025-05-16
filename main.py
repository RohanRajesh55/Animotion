import cv2
import time
import logging
from tkinter import Tk, Label, StringVar, Frame
from PIL import Image, ImageTk

from detectors.facial_landmarks_processor import FacialLandmarksProcessor
from emotion_recognition import recognize_emotion
from utils.shared_variables import SharedVariables

# Configure logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Detect GPU availability based on OpenCV CUDA capabilities.
# (Note: This works only if your OpenCV build supports CUDA.)
has_gpu = False
try:
    if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        has_gpu = True
        logger.info("GPU detected – utilizing GPU acceleration if available.")
    else:
        logger.info("No GPU detected – running on CPU.")
except Exception as e:
    logger.info("No GPU detected - error when checking GPU: %s", e)

# Set parameters based on GPU availability.
if has_gpu:
    # If GPU is available, attempt to update more frequently.
    VIDEO_UPDATE_DELAY = 10        # Delay in ms (targeting a higher update rate)
    SKIP_PROCESSING_FRAMES = 1       # Process every frame (or reduce skipping)
else:
    # If working on CPU only, skip heavy processing on more frames
    # to try to maintain higher frame rates (aiming closer to 30 FPS).
    VIDEO_UPDATE_DELAY = 15        # Delay in ms; lower delay may try to increase FPS
    SKIP_PROCESSING_FRAMES = 4     # Skip heavy processing on most frames

# Constants for detections.
BLINK_THRESHOLD = 0.25        # When average EAR is below this, a blink is detected.
MOUTH_OPEN_THRESHOLD = 0.5     # MAR (or lip sync value) above this indicates mouth open.
EMOTION_INTERVAL = 2.0         # Run emotion recognition at most once every 2 seconds.
FAILED_FRAME_THRESHOLD = 10    # Number of consecutive failed frame reads before reinitializing the camera.

class AnimotionInterface:
    def __init__(self, video_source: int = 0) -> None:
        """
        Initialize the Animotion interface: set up video capture, processing, and build the Tkinter UI.
        """
        self.video_source = video_source
        self.shared_vars = SharedVariables()
        self.processor = FacialLandmarksProcessor()

        # Initialize VideoCapture.
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            logger.error("Cannot open video source: %s", self.video_source)
            raise Exception("Video capture initialization failed.")

        # (Optional) Set a fixed, lower resolution to help performance.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize UI.
        self.root = Tk()
        self.root.title("Animotion - Facial Tracking Interface")

        # Bind the "q" key to quit.
        self.root.bind("<Key>", self.on_key_press)

        # Video display.
        self.video_label = Label(self.root)
        self.video_label.pack()

        # Status frame.
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

        # Variables for managing processing.
        self.running = False
        self.failed_frame_count = 0
        self.last_emotion_time = 0.0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.skip_counter = 0

    def on_key_press(self, event):
        """If 'q' is pressed, stop processing and close the window."""
        if event.char.lower() == 'q':
            self.stop()
            self.root.destroy()

    def start(self) -> None:
        """Begin the video updating loop immediately."""
        if self.running:
            return
        self.running = True
        self.last_emotion_time = time.time()
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.skip_counter = 0
        self.update_video()  # Start updating frames.
        logger.info("Animotion interface started.")

    def stop(self) -> None:
        """Stop the video updating loop."""
        self.running = False
        self.status_var.set("Stopped")
        logger.info("Animotion interface stopped.")

    def reinitialize_video_capture(self) -> None:
        """Reinitialize VideoCapture in case of repeated frame capture failures."""
        try:
            self.cap.release()
        except Exception as e:
            logger.error("Error releasing VideoCapture: %s", e)
        logger.info("Reinitializing video capture on source %s...", self.video_source)
        self.cap = cv2.VideoCapture(self.video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            logger.error("Reinitialization failed; video source %s cannot be opened.", self.video_source)
        else:
            logger.info("Video capture reinitialized successfully.")

    def update_video(self) -> None:
        """
        Capture a frame, perform (optional) heavy processing every N frames,
        update the FPS and detections, and refresh the display.
        """
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.failed_frame_count += 1
            logger.warning("Failed to capture frame. Count: %s", self.failed_frame_count)
            if self.failed_frame_count >= FAILED_FRAME_THRESHOLD:
                logger.warning("Failed frame threshold reached. Reinitializing capture.")
                self.reinitialize_video_capture()
                self.failed_frame_count = 0
            self.root.after(VIDEO_UPDATE_DELAY, self.update_video)
            return
        else:
            self.failed_frame_count = 0

        self.frame_count += 1
        self.skip_counter += 1
        current_time = time.time()

        # Perform heavy processing (landmark detection, emotion recognition) only on selected frames.
        if self.skip_counter % SKIP_PROCESSING_FRAMES == 0:
            metrics = self.processor.process_frame(frame)
            if metrics:
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

            # Run emotion recognition every EMOTION_INTERVAL seconds.
            if current_time - self.last_emotion_time > EMOTION_INTERVAL:
                try:
                    emotion = recognize_emotion(frame)
                except Exception as e:
                    logger.error("Emotion recognition failed: %s", e)
                    emotion = "unknown"
                self.shared_vars.emotion = emotion
                self.last_emotion_time = current_time

        # Update detection status text.
        detection_text = ""
        if self.shared_vars.ear_left is not None and self.shared_vars.ear_right is not None:
            avg_ear = (self.shared_vars.ear_left + self.shared_vars.ear_right) / 2.0
            blink_det = "Yes" if avg_ear < BLINK_THRESHOLD else "No"
            detection_text += f"Blink: {blink_det}\n"
        else:
            detection_text += "Blink: N/A\n"

        if self.shared_vars.mar is not None:
            mouth_status = "Yes" if self.shared_vars.mar > MOUTH_OPEN_THRESHOLD else "No"
            detection_text += f"Mouth Open: {mouth_status}\n"
        else:
            detection_text += "Mouth Open: N/A\n"

        emotion_disp = self.shared_vars.emotion if self.shared_vars.emotion is not None else "N/A"
        detection_text += f"Emotion: {emotion_disp}"
        self.detection_status_var.set(detection_text)

        # Calculate FPS every second.
        elapsed_time = current_time - self.last_fps_time
        if elapsed_time >= 1.0:
            fps = self.frame_count / elapsed_time
            self.fps_var.set(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_fps_time = current_time

        # Convert the frame from BGR to RGB, then display it.
        try:
            cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # Keep reference.
            self.video_label.configure(image=imgtk)
        except Exception as e:
            logger.error("Error converting frame for display: %s", e)

        # Schedule the next update.
        self.root.after(VIDEO_UPDATE_DELAY, self.update_video)

    def run(self) -> None:
        """Start processing and enter the Tkinter mainloop."""
        self.start()
        self.root.mainloop()

def main() -> None:
    interface = AnimotionInterface(video_source=0)
    interface.run()

if __name__ == '__main__':
    main()