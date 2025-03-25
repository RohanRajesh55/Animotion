import cv2
import numpy as np

class HeadPoseEstimator:
    def __init__(self, frame_width, frame_height):
        """
        Initialize the head pose estimator with a camera matrix and predefined 3D model points.
        """
        self.camera_matrix = np.array([
            [frame_width, 0, frame_width / 2],
            [0, frame_height, frame_height / 2],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # More precise 3D model points (better alignment with actual face landmarks)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -60.0, -25.0),      # Chin
            (-30.0, -35.0, -20.0),    # Left eye left corner
            (30.0, -35.0, -20.0),     # Right eye right corner
            (-25.0, 30.0, -20.0),     # Left mouth corner
            (25.0, 30.0, -20.0)       # Right mouth corner
        ], dtype="double")

    def get_head_pose(self, image_points):
        """
        Estimate head pose using solvePnP.

        :param image_points: 2D image points from facial landmarks.
        :return: (rotation_vector, translation_vector, euler_angles)
        """
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, None, None  # Avoid errors if PnP fails

        # Convert rotation vector to Euler angles
        euler_angles = self.get_euler_angles(rotation_vector)

        return rotation_vector, translation_vector, euler_angles

    def get_euler_angles(self, rotation_vector):
        """
        Convert rotation vector to Euler angles (yaw, pitch, roll).

        :param rotation_vector: Rotation vector from solvePnP.
        :return: (yaw, pitch, roll) in degrees.
        """
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        projection_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
        
        yaw, pitch, roll = euler_angles.flatten()
        return (yaw, pitch, roll)
