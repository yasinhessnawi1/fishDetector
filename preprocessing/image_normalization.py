import numpy as np

def normalize_image(image, scale='0-1'):
    """
    Normalize the pixel values of an image to a standard scale.

    Parameters:
    - image: The input image as a NumPy array.
    - scale: The scale to normalize the pixel values to.
             '0-1' for a scale of 0 to 1, and '-1-1' for a scale of -1 to 1.

    Returns:
    - normalized_image: The image with normalized pixel values.
    """
    # Convert image to float type for precise division operation
    image = image.astype(np.float32)

    if scale == '0-1':
        # Normalize pixel values to [0, 1]
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    elif scale == '-1-1':
        # Normalize pixel values to [-1, 1]
        normalized_image = (2 * (image - np.min(image)) / (np.max(image) - np.min(image))) - 1
    else:
        raise ValueError("Invalid scale value. Choose either '0-1' or '-1-1'.")

    return normalized_image
