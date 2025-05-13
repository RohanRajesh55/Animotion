"""
head pose estimator using OpenCV's solvePnP.

This module defines a constant 3D model of facial landmarks and provides a function
to estimate head pose and optionally extract Euler angles.
"""

import logging
import cv2
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Default 3D model points for facial landmarks (assumed units: millimeters)
DEFAULT_MODEL_POINTS: np.ndarray = np.array([
    (0.0, 0.0, 0.0),              # Nose tip
    (0.0, -100.0, -30.0),         # Chin
    (-70.0, -70.0, -50.0),        # Left eye corner
    (70.0, -70.0, -50.0),         # Right eye corner
    (-60.0, 50.0, -50.0),         # Left mouth corner
    (60.0, 50.0, -50.0)           # Right mouth corner
], dtype="double")


def get_head_pose(image_points: np.ndarray,
                  camera_matrix: np.ndarray,
                  dist_coeffs: np.ndarray,
                  model_points: Optional[np.ndarray] = None,
                  return_euler: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Estimate the head pose using 2D facial landmarks and solvePnP.

    Args:
        image_points (np.ndarray): Array of shape (6, 2) with 2D facial landmark coordinates.
        camera_matrix (np.ndarray): 3x3 intrinsic camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients (set to zeros if undistorted).
        model_points (Optional[np.ndarray]): Optional override for default 3D model points.
        return_euler (bool): If True, returns Euler angles (yaw, pitch, roll).

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
            - rotation_vector (3x1)
            - translation_vector (3x1)
            - euler_angles (3x1) if return_euler is True, else None

    Raises:
        ValueError: If input shape is invalid or solvePnP fails.
    """
    model_points = model_points if model_points is not None else DEFAULT_MODEL_POINTS

    if image_points.shape != (model_points.shape[0], 2):
        logger.error("Invalid image_points shape: expected (%d, 2), got %s",
                     model_points.shape[0], image_points.shape)
        raise ValueError("Invalid image_points shape.")

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        logger.error("cv2.solvePnP failed to estimate head pose.")
        raise ValueError("Head pose estimation failed.")

    euler_angles = None
    if return_euler:
        try:
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            euler_angles = euler_angles[:3]  # Extract yaw, pitch, roll
        except Exception as e:
            logger.warning("Euler angle extraction failed: %s", e)
            euler_angles = None

    return rotation_vector, translation_vector, euler_angles
