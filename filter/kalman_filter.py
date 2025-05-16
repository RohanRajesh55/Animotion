import numpy as np
import logging

logger = logging.getLogger(__name__)

class KalmanFilter:
    """
    A 1-dimensional Kalman Filter for smoothing noisy signals.

    Attributes:
        process_variance (float): Variance of the process noise.
        measurement_variance (float): Variance of the measurement noise.
        estimated_error (float): Current uncertainty in the state estimate.
        state_estimate (float): Current estimated state.
        kalman_gain (float): The computed Kalman gain.
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        estimated_error: float = 1.0
    ) -> None:
        """
        Initialize the Kalman Filter with the specified process and measurement noise characteristics.

        Args:
            process_variance (float): Variance of the process noise.
            measurement_variance (float): Variance of the measurement noise.
            estimated_error (float): Initial uncertainty in the state estimate.
        """
        self.process_variance: float = process_variance
        self.measurement_variance: float = measurement_variance
        self.estimated_error: float = estimated_error
        self.state_estimate: float = 0.0
        self.kalman_gain: float = 0.0

        logger.debug(f"Initialized KalmanFilter with process_variance={self.process_variance}, "
                     f"measurement_variance={self.measurement_variance}, "
                     f"estimated_error={self.estimated_error}")

    def update(self, measurement: float) -> float:
        """
        Update the Kalman filter with a new measurement. This method performs:
        
          - A prediction update: Increases the uncertainty by the process variance.
          - A correction update: Computes the Kalman gain, refines the state estimate, and
            reduces the uncertainty based on the measurement.

        Args:
            measurement (float): The new measurement value.

        Returns:
            float: The updated state estimate.
        """
        # Prediction update: Increase uncertainty.
        self.estimated_error += self.process_variance
        logger.debug(f"After prediction: estimated_error = {self.estimated_error}")

        # Compute Kalman gain.
        self.kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        logger.debug(f"Calculated kalman_gain = {self.kalman_gain}")

        # Correction update: Adjust the state estimate based on the new measurement.
        self.state_estimate += self.kalman_gain * (measurement - self.state_estimate)
        logger.debug(f"Updated state_estimate = {self.state_estimate}")

        # Update error estimate: Reduce the uncertainty after correction.
        self.estimated_error *= (1 - self.kalman_gain)
        logger.debug(f"Updated estimated_error after correction = {self.estimated_error}")

        return self.state_estimate

    def reset(self, initial_state: float = 0.0, initial_error: float = 1.0) -> None:
        """
        Reset the filter to a specified initial state and uncertainty.

        Args:
            initial_state (float): The value to reset the state estimate to (default 0.0).
            initial_error (float): The value to reset the estimation error to (default 1.0).
        """
        self.state_estimate = initial_state
        self.estimated_error = initial_error
        self.kalman_gain = 0.0
        logger.debug(f"KalmanFilter reset to state_estimate = {self.state_estimate}, "
                     f"estimated_error = {self.estimated_error}")

    def __repr__(self) -> str:
        return (f"KalmanFilter(process_variance={self.process_variance}, "
                f"measurement_variance={self.measurement_variance}, "
                f"estimated_error={self.estimated_error}, "
                f"state_estimate={self.state_estimate}, "
                f"kalman_gain={self.kalman_gain})")