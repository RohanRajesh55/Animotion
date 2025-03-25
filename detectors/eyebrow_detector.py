import numpy as np
from collections import deque
from utils.calculations import calculate_distance_coords

class EyebrowDetector:
    def __init__(self, raise_threshold=1.2, smoothing_frames=3):
        """
        Eyebrow Detector for detecting eyebrow movement using Eyebrow Raise Ratio (EBR).

        :param raise_threshold: Threshold for detecting eyebrow raise.
        :param smoothing_frames: Number of frames for EBR smoothing.
        """
        self.raise_threshold = raise_threshold
        self.ebr_history = deque(maxlen=smoothing_frames)  # Efficient rolling window

    def calculate_ebr(self, eyebrow_landmarks, eye_landmarks, width, height):
        """
        Calculate Eyebrow Raise Ratio (EBR) for eyebrow movement detection.

        :param eyebrow_landmarks: List of eyebrow landmarks (left or right).
        :param eye_landmarks: List of eye landmarks (left or right).
        :param width: Width of the frame.
        :param height: Height of the frame.
        :return: Smoothed Eyebrow Raise Ratio.
        """
        if len(eyebrow_landmarks) < 2 or len(eye_landmarks) < 4:
            return 0  # Invalid landmarks
        
        # Convert landmarks to 2D coordinates
        eyebrow_coord = (int(eyebrow_landmarks[1].x * width), int(eyebrow_landmarks[1].y * height))
        eye_coord = (int(eye_landmarks[1].x * width), int(eye_landmarks[1].y * height))

        # Calculate vertical distance between eyebrow and eye
        vertical_distance = calculate_distance_coords(eyebrow_coord, eye_coord)

        # Calculate eye width
        left_eye = (int(eye_landmarks[0].x * width), int(eye_landmarks[0].y * height))
        right_eye = (int(eye_landmarks[3].x * width), int(eye_landmarks[3].y * height))
        eye_width = calculate_distance_coords(left_eye, right_eye)

        if eye_width == 0:
            return 0  # Prevent division by zero

        # Compute Eyebrow Raise Ratio (EBR)
        ebr = vertical_distance / eye_width

        # Apply moving average smoothing
        self.ebr_history.append(ebr)
        return np.mean(self.ebr_history)  # Smoothed EBR

    def is_raised(self, ebr, threshold=None):
        """
        Detects if the eyebrows are raised based on threshold.

        :param ebr: Eyebrow Raise Ratio.
        :param threshold: Custom threshold (optional), defaults to instance threshold.
        :return: True if eyebrows are raised, else False.
        """
        return ebr > (threshold if threshold is not None else self.raise_threshold)
