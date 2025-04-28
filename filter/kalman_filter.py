import numpy as np

class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2, estimated_error=1.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_error = estimated_error
        self.state_estimate = 0.0
        self.kalman_gain = 0.0

    def update(self, measurement):
        # Prediction update
        self.estimated_error += self.process_variance

        # Measurement update (correction)
        self.kalman_gain = self.estimated_error / (self.estimated_error + self.measurement_variance)
        self.state_estimate += self.kalman_gain * (measurement - self.state_estimate)
        self.estimated_error *= (1 - self.kalman_gain)

        return self.state_estimate
