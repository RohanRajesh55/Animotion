# detectors/head_pose_estimator.py

import cv2
import numpy as np

def get_head_pose(image_points, camera_matrix, dist_coeffs):
    """
    Calculate head pose using solvePnP.

    :param image_points: 2D image points from facial landmarks
    :param camera_matrix: Camera matrix computed from frame dimensions
    :param dist_coeffs: Distortion coefficients (assumed zero here)
    :return: rotation_vector, translation_vector
    """
    # Define the 3D model points of facial landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -100.0, -30.0),        # Chin
        (-70.0, -70.0, -50.0),       # Left eye corner
        (70.0, -70.0, -50.0),        # Right eye corner
        (-60.0, 50.0, -50.0),        # Left mouth corner
        (60.0, 50.0, -50.0)          # Right mouth corner
    ])

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise ValueError("Could not solve PnP")

    return rotation_vector, translation_vector
