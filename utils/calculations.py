import numpy as np
import time
from typing import Tuple


def calculate_distance_coords(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2D coordinates.
    """
    return np.hypot(point2[0] - point1[0], point2[1] - point1[1])


def fps_calculation(frame_count: int, start_time: float, use_perf_counter: bool = False) -> Tuple[int, float]:
    """
    Calculate the current frames per second (FPS).

    Parameters:
        frame_count (int): Frames processed so far.
        start_time (float): Timestamp when processing started.
        use_perf_counter (bool): Whether to use `perf_counter()` instead of `time()`.

    Returns:
        Tuple[int, float]: (updated frame_count, calculated FPS)
    """
    frame_count += 1
    now = time.perf_counter() if use_perf_counter else time.time()
    elapsed_time = now - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
    return frame_count, fps
