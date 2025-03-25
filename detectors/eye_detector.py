import numpy as np
from collections import deque
from utils.calculations import calculate_distance_coords

class EyeDetector:
    def __init__(self, ear_threshold=0.25, smoothing_frames=3):
        """
        Eye Detector class for calculating Eye Aspect Ratio (EAR) and detecting blinks.

        :param ear_threshold: Threshold for blink detection (default 0.25).
        :param smoothing_frames: Number of frames for EAR smoothing.
        """
        self.ear_threshold = ear_threshold
        self.ear_history = deque(maxlen=smoothing_frames)  # Efficient rolling window

    def calculate_ear(self, eye_landmarks, width, height):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.

        :param eye_landmarks: List of eye landmarks from Mediapipe or dlib.
        :param width: Width of the frame.
        :param height: Height of the frame.
        :return: Smoothed Eye Aspect Ratio (EAR).
        """
        if len(eye_landmarks) != 6:
            raise ValueError("Eye landmarks should contain exactly 6 points.")

        # Convert landmarks to 2D coordinates
        coords = [(int(pt.x * width), int(pt.y * height)) for pt in eye_landmarks]

        # Calculate vertical distances
        vertical1 = calculate_distance_coords(coords[1], coords[5])  # p2 - p6
        vertical2 = calculate_distance_coords(coords[2], coords[4])  # p3 - p5

        # Calculate horizontal eye distance
        horizontal = calculate_distance_coords(coords[0], coords[3])  # p1 - p4

        if horizontal == 0:
            return 0  # Prevent division by zero

        # Compute EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)

        # Apply moving average smoothing
        self.ear_history.append(ear)
        return np.mean(self.ear_history)  # Smoothed EAR

    def is_blinking(self, ear, threshold=None):
        """
        Detects if the user is blinking based on EAR threshold.

        :param ear: Eye Aspect Ratio.
        :param threshold: Custom threshold (optional), defaults to instance EAR threshold.
        :return: True if blink detected, else False.
        """
        return ear < (threshold if threshold is not None else self.ear_threshold)
