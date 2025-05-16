import cv2
import logging
import time
import threading
from typing import Optional

from emotion_recognition import recognize_emotion
from utils.shared_variables import SharedVariables

logger = logging.getLogger(__name__)

def update_emotion(shared_vars: SharedVariables, video_source: int = 0) -> None:
    """
    Continuously capture video frames from the specified source, run emotion recognition,
    and update the SharedVariables instance with the latest detected emotion.

    This function runs as a dedicated thread and updates shared_vars.emotion every few seconds.
    
    Args:
        shared_vars (SharedVariables): The shared variables container to update with emotion.
        video_source (int): Index for the video capture source (default is 0).
    """
    cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Unable to open camera for emotion recognition.")
        return

    logger.info("Starting emotion recognition thread.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to capture frame for emotion recognition.")
            time.sleep(0.1)
            continue

        # Recognize emotion in the current frame.
        emotion = recognize_emotion(frame)
        logger.debug(f"Detected emotion: {emotion}")

        # Update the shared emotion value.
        shared_vars.emotion = emotion

        # Sleep briefly (e.g., update every 2 seconds).
        time.sleep(2)

    cap.release()

def start_emotion_integration(shared_vars: SharedVariables, video_source: int = 0) -> threading.Thread:
    """
    Start the emotion recognition integration in its own dedicated thread.
    
    Args:
        shared_vars (SharedVariables): Shared variables container to update.
        video_source (int): Video source index (default 0).
    
    Returns:
        threading.Thread: The thread running the emotion update loop.
    """
    emotion_thread = threading.Thread(target=update_emotion, args=(shared_vars, video_source), daemon=True)
    emotion_thread.start()
    logger.info("Emotion recognition thread started.")
    return emotion_thread

if __name__ == "__main__":
    # Example production-level startup code:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shared_vars = SharedVariables()
    start_emotion_integration(shared_vars)
    
    # The main thread could perform other tasks (e.g., facial landmark processing, WebSocket updates)
    try:
        while True:
            logging.info(f"Current emotion: {shared_vars.emotion}")
            time.sleep(5)
    except KeyboardInterrupt:
        logging.info("Emotion integration interrupted. Exiting...")