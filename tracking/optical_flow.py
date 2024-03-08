import cv2
import numpy as np

def lucas_kanade_optical_flow(prev_frame, next_frame, prev_points):
    """
    Track given points from the previous frame to the next frame using the Lucas-Kanade Optical Flow method.

    Parameters:
    - prev_frame: The previous frame in the video sequence.
    - next_frame: The next frame in the video sequence.
    - prev_points: Points to track from the previous frame.

    Returns:
    - next_points: The tracked points in the next frame.
    - status: Array indicating the status of tracked points (1 if successfully tracked).
    """
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_points, None, **lk_params)
    return next_points, status
