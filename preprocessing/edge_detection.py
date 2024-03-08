import cv2

def apply_canny_edge_detector(image, low_threshold=50, high_threshold=150):
    """
    Apply the Canny edge detector to an input image.

    Parameters:
    - image: The input image on which edge detection will be performed.
    - low_threshold: The lower bound for detecting strong edges.
    - high_threshold: The upper bound for detecting strong edges.

    Returns:
    - edges: The binary image with edges marked as white (255) and non-edges as black (0).
    """
    # Convert the image to grayscale as Canny edge detector requires grayscale inputs
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply the Canny edge detector
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)

    return edges
