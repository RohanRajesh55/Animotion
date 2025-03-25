import time
import numpy as np

def fps_calculation(frame_count, start_time):
    """Calculates FPS efficiently."""
    frame_count += 1
    elapsed_time = time.time() - start_time
    return frame_count, round(frame_count / elapsed_time, 2) if elapsed_time > 0 else 0.0

def adaptive_threshold(history, base_threshold, min_limit, factor=0.85):
    """Adaptive threshold calculation for better tracking stability."""
    if len(history) > 50:
        history.pop(0)
    return max(min(history) * factor, min_limit) if history else base_threshold
