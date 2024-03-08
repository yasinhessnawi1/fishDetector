import cv2
import numpy as np

class KalmanFilterTracker:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1, error=1):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * error

    def update(self, measurement):
        self.kalman.correct(measurement)
        return self.kalman.predict()
