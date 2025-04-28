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
    return np.hypot(point2[0] - point1[0], point2[1] - point1[1])

def fps_calculation(frame_count, start_time):
    """
    Calculate the frames per second (FPS)

    :param frame_count: Current frame count
    :param start_time: Time when the frame processing started
    :return: Updated frame count, current FPS
    """
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return frame_count, fps
