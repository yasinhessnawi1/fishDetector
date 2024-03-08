import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, initial_state):
        """
        Initialize the object tracker with the initial state.

        Parameters:
        - initial_state: The initial state of the object to track, given as [x, y, vx, vy],
                          where (x, y) is the initial position and (vx, vy) is the initial velocity.
        """
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, 4) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, 2) * 0.5
        self.kalman.statePre = np.array(initial_state, np.float32)

    def update(self, measurement):
        """
        Update the tracker based on the new measurement.

        Parameters:
        - measurement: The new measurement of the object's position, given as [x, y].

        Returns:
        - predicted_state: The predicted state after the update, given as [x, y, vx, vy].
        """
        corrected_state = self.kalman.correct(np.array(measurement, np.float32))
        predicted_state = self.kalman.predict()

        return predicted_state

    def predict(self):
        """
        Predict the next state of the object without a new measurement.

        Returns:
        - predicted_state: The predicted state, given as [x, y, vx, vy].
        """
        predicted_state = self.kalman.predict()

        return predicted_state
