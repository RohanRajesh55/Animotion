import logging
from typing import Union

logger = logging.getLogger(__name__)


class KalmanFilter:
    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        initial_estimate: float = 0.0,
        initial_error: float = 1.0,
    ):
        """
        Initialize the Kalman filter.
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state_estimate = initial_estimate
        self.estimated_error = initial_error
        self.kalman_gain = 0.0

    def predict(self) -> None:
        """
        Predict phase: update internal error estimate by adding process noise.
        (Useful if separating predict/correct logic externally)
        """
        self.estimated_error += self.process_variance

    def update(self, measurement: Union[int, float]) -> float:
        """
        Perform one update with the given measurement.
        """
        if not isinstance(measurement, (int, float)):
            logger.error("Invalid measurement type: expected int or float, got %s", type(measurement))
            raise ValueError("Measurement must be numeric.")

        self.predict()

        # Kalman gain
        self.kalman_gain = self.estimated_error / (
            self.estimated_error + self.measurement_variance
        )

        # Update estimate with new measurement
        self.state_estimate += self.kalman_gain * (measurement - self.state_estimate)

        # Update error estimate
        self.estimated_error *= (1 - self.kalman_gain)

        return self.state_estimate

    def reset(self, initial_estimate: float = 0.0, initial_error: float = 1.0) -> None:
        """
        Reset the filter to initial state.
        """
        self.state_estimate = initial_estimate
        self.estimated_error = initial_error
        self.kalman_gain = 0.0

    def get_state(self) -> float:
        """
        Get the current state estimate.
        """
        return self.state_estimate
