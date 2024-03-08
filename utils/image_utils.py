import cv2

class ImageUtils:
    @staticmethod
    def convert_to_grayscale(image):
        """
        Convert an input image to grayscale.

        Parameters:
        - image: The input color image.

        Returns:
        - grayscale_image: The converted grayscale image.
        """
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
        """
        Apply Gaussian blur to an image to reduce noise.

        Parameters:
        - image: The input image to be blurred.
        - kernel_size: The size of the Gaussian kernel.
        - sigma: The standard deviation of the Gaussian kernel. If 0, it is calculated from the kernel size.

        Returns:
        - blurred_image: The blurred image.
        """
        blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
        return blurred_image
