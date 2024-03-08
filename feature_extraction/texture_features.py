import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def extract_lbp_features(image, P=8, R=1, method='uniform'):
    """
    Extract Local Binary Pattern (LBP) features from an image.

    Parameters:
    - image: The input image (preferably in grayscale).
    - P: Number of circularly symmetric neighbor set points (default is 8).
    - R: Radius of circle (default is 1).
    - method: Method to extract LBP features. 'uniform' patterns are a good choice.

    Returns:
    - lbp_image: Image transformed into LBP space.
    - lbp_hist: Histogram of LBP features.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the LBP operation
    lbp_image = local_binary_pattern(image, P, R, method)

    # Calculate the histogram of the LBP result
    n_bins = int(lbp_image.max() + 1)
    lbp_hist, _ = np.histogram(lbp_image.ravel(), density=True, bins=n_bins, range=(0, n_bins))

    return lbp_image, lbp_hist
