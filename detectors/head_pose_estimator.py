import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_head_pose(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate head pose (rotation and translation vectors) using the solvePnP algorithm.

    The function uses predefined 3D facial model points corresponding to the following landmarks:
        - Nose tip
        - Chin
        - Left eye corner
        - Right eye corner
        - Left mouth corner
        - Right mouth corner

    It solves for the head pose by mapping these 3D model points to the provided 2D image points.

    :param image_points: A NumPy array of shape (6, 2) containing the 2D image coordinates of facial landmarks.
    :param camera_matrix: A 3x3 NumPy array representing the camera intrinsic matrix.
    :param dist_coeffs: A NumPy array containing the distortion coefficients.
    :return: A tuple (rotation_vector, translation_vector) as NumPy arrays.
    :raises ValueError: If the PnP solution fails.
    """
    # Define the 3D model points corresponding to the facial landmarks.
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -100.0, -30.0),       # Chin
        (-70.0, -70.0, -50.0),      # Left eye corner
        (70.0, -70.0, -50.0),       # Right eye corner
        (-60.0, 50.0, -50.0),       # Left mouth corner
        (60.0, 50.0, -50.0)         # Right mouth corner
    ], dtype=np.float64)

    # Compute head pose using solvePnP.
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        logger.error("Head pose estimation failed: Could not solve PnP.")
        raise ValueError("Could not solve PnP for head pose estimation.")

    return rotation_vector, translation_vector