import cv2
import numpy as np


def apply_otsu_thresholding(image):
    """
    Apply Otsu's thresholding method to an input image to perform segmentation.

    Parameters:
    - image: The input image on which segmentation will be performed.

    Returns:
    - segmented_image: The image after applying Otsu's thresholding, where the foreground is white and the background is black.
    """
    # Convert the image to grayscale, as Otsu's method works on single channel images
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve thresholding
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Otsu's thresholding
    _, segmented_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return segmented_image
