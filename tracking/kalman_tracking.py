import cv2
import numpy as np

class KalmanTracker:
    def __init__(self, state_size, meas_size, contr_size, type=cv2.CV_32F):
        # Create the Kalman Filter
        self.kalman = cv2.KalmanFilter(state_size, meas_size, contr_size, type)

        # State transition matrix (A)
        self.kalman.transitionMatrix = np.eye(state_size, dtype=np.float32)

        # Measurement matrix (H)
        self.kalman.measurementMatrix = np.eye(meas_size, state_size, dtype=np.float32)

        # Process noise covariance (Q)
        self.kalman.processNoiseCov = np.eye(state_size, dtype=np.float32) * 1e-2

        # Measurement noise covariance (R)
        self.kalman.measurementNoiseCov = np.eye(meas_size, dtype=np.float32) * 1e-1

        # Error covariance matrix (P)
        self.kalman.errorCovPost = np.eye(state_size, dtype=np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        return self.kalman.correct(measurement)

    def update(self, measurement):
        self.predict()
        return self.correct(measurement)
