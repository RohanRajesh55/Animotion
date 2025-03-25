# utils/calculations.py

import numpy as np
import time

def calculate_distance_coords(point1, point2):
    """
    Calculate Euclidean distance between two 2D coordinates.

    :param point1: Tuple (x1, y1)
    :param point2: Tuple (x2, y2)
    :return: Distance between point1 and point2
    """
    if not point1 or not point2:
        return 0.0  # Ensure valid points

    return np.linalg.norm(np.array(point1) - np.array(point2))

class FPSCounter:
    """
    A class to calculate FPS using a moving average to smooth variations.
    """
    def __init__(self, smoothing=10):
        self.frame_times = []
        self.smoothing = max(1, smoothing)  # Ensure smoothing is at least 1

    def update(self):
        """
        Updates the frame count and computes FPS.

        :return: Smoothed FPS value
        """
        current_time = time.time()
        self.frame_times.append(current_time)

        # Keep only the last `smoothing` frame times
        if len(self.frame_times) > self.smoothing:
            self.frame_times.pop(0)

        return self.get_fps()

    def reset(self):
        """Resets the frame times list."""
        self.frame_times.clear()

    def get_fps(self):
        """
        Returns the current FPS value.
        """
        if len(self.frame_times) < 2:
            return 0.0  # Avoid division by zero

        return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

    def get_avg_fps(self):
        """
        Returns the smoothed average FPS value.
        """
        return self.get_fps()
