import time
import threading
import logging
from typing import Any
from utils.calculations import fps_calculation

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    PerformanceMonitor calculates and updates the FPS value for the application.
    It uses the fps_calculation utility and updates the 'fps' attribute in the shared variables.
    """

    def __init__(self, shared_vars: Any, update_interval: float = 1.0) -> None:
        """
        Initializes the performance monitor.

        Args:
            shared_vars (Any): The SharedVariables instance where the FPS is stored.
            update_interval (float): Time interval in seconds to update FPS (default: 1.0).
        """
        self.shared_vars = shared_vars
        self.update_interval = update_interval
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the performance monitoring in a separate daemon thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Performance monitor started.")

    def stop(self) -> None:
        """Stop the performance monitoring loop and join the thread."""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Performance monitor stopped.")

    def _monitor_loop(self) -> None:
        """
        Monitoring loop that updates the shared FPS value every `update_interval` seconds.
        It uses the fps_calculation function to compute the current FPS based on the frame count and elapsed time.
        """
        frame_count = 0
        start_time = time.time()

        while self.running:
            # Sleep for the designated interval.
            time.sleep(self.update_interval)
            
            # Increment the frame count.
            frame_count += 1

            # Compute FPS using current frame count and elapsed time.
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps, frame_count = fps_calculation(frame_count, start_time)
                self.shared_vars.fps = fps
                logger.info(f"FPS updated: {fps:.2f}")
                
                # Reset for the next monitoring window.
                start_time = time.time()