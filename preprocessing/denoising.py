import cv2

class ImageDenoiser:
    """
    A class for denoising images using the Fast Non-Local Means Denoising algorithm.
    """

    def __init__(self, h=10, template_window_size=7, search_window_size=21):
        """
        Initializes the ImageDenoiser with specific parameters for the denoising algorithm.

        Parameters:
        - h: The parameter regulating filter strength. Higher h value removes noise better but also removes image details. Lower h value preserves details but also preserves some noise.
        - template_window_size: Size in pixels of the template patch used for denoising. Should be odd.
        - search_window_size: Size in pixels of the window used to search for patches similar to the template patch. Should be odd.
        """
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def denoise(self, image):
        """
        Applies the Fast Non-Local Means Denoising algorithm to the input image.

        Parameters:
        - image: The noisy input image to be denoised.

        Returns:
        - denoised_image: The denoised output image.
        """
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, self.h, self.h, self.template_window_size, self.search_window_size)
        return denoised_image


def fast_non_local_means_denoising(image, h=10, h_for_color_components=10, template_window_size=7,
                                   search_window_size=21):
    """
    Apply Fast Non-Local Means Denoising algorithm to an image.

    Parameters:
    - image: The input noisy image.
    - h: Filter strength for the luminance component. The larger the h, the smoother the image (but more details are removed).
    - h_for_color_components: Same as h, but for color images only.
    - template_window_size: Odd size of the window used to compute the weighted average. It should be around 7.
    - search_window_size: Odd size of the window used to search for patches that look like the template. It should be around 21.

    Returns:
    - denoised_image: The denoised image.
    """
    # Check if the image is grayscale or color
    if len(image.shape) == 2:
        # Grayscale image
        denoised_image = cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
    elif len(image.shape) == 3:
        # Color image
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, h_for_color_components, template_window_size,
                                                         search_window_size)
    else:
        raise ValueError("The input image must be either a 2D grayscale or a 3D color image.")

    return denoised_image