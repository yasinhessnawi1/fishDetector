import cv2
import numpy as np


def compute_hog_features(image):
    """
    Compute the Histogram of Oriented Gradients (HOG) features for a given image.

    Parameters:
    - image: The input image for which HOG features are to be computed.

    Returns:
    - hog_features: The HOG feature vector for the input image.
    """
    # Convert image to grayscale as HOG needs single channel image input
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define HOG Descriptor parameters
    # can be adjsted!!!
    win_size = (64, 64)  # Size of the detection window
    block_size = (16, 16)  # Size of blocks
    block_stride = (8, 8)  # Block stride
    cell_size = (8, 8)  # Size of cells
    nbins = 9  # Number of bins for the histogram

    # Initialize HOG Descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # Compute HOG features
    hog_features = hog.compute(gray_image)

    # Return the computed HOG features
    return hog_features.flatten()  # Flatten the feature vector for easier handling
