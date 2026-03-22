import math
import numpy as np

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma:float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.

    :param lowImage: the image to low-pass filter
           (either greyscale shape=(rows,cols) or
           colour shape=(rows,cols,channels)).
    :type numpy.ndarray

    :param lowSigma: the standard deviation of the Gaussian used for low-pass
           filtering lowImage.
    :type float

    :param highImage: the image to high-pass filter
           (either greyscale shape=(rows,cols) or
           colour shape=(rows,cols,channels)).
    :type numpy.ndarray

    :param highSigma: the standard deviation of the Gaussian used for low-pass
           filtering highImage before subtraction to create the high-pass
           filtered image.
    :type float

    :returns returns the hybrid image created by low-pass filtering lowImage
             with a Gaussian s.d. lowSigma and combining it with a high-pass
             image created by subtracting highImage from highImage convolved
             with a Gaussian s.d. highSigma. The resultant image has
             the same size as the input images.
    :rtype numpy.ndarray
    """

    low_pass = convolve(image=lowImage, kernel=makeGaussianKernel(lowSigma))
    # Create a high-pass by subtracting a low-pass from itself.
    high_pass = highImage - convolve(image=highImage, kernel=makeGaussianKernel(highSigma))

    # Make the images the same size
    low_resized, high_resized = resize_images(low_pass, high_pass)

    # Create the hybrid image
    hybrid = low_resized + high_resized

    # Remap output to range [0, 255]
    min_range, max_range = np.min(hybrid), np.max(hybrid)
    output = (hybrid - min_range) / (max_range - min_range) * 255

    return output.astype(np.uint8)

def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    size = int(8. * sigma + 1.)
    if size % 2 == 0: size += 1
    x = np.ndarray((size,))
    for idx, j in enumerate(range(int(-((size-1)/2)), int((size-1)/2)+1)):
        x[idx] = math.e ** ((-0.5 * j ** 2) / sigma ** 2)
    kernel = np.asarray([[i * j for i in x] for j in x])

    return (1/np.sum(kernel)) * kernel

def resize_images(low_pass:np.ndarray, high_pass:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize images so that they can be combined into a hybrid image.
    Images will be zero-padded so that they are in the centre of the resulting image.
    The resulting image will be of shape (max(low_pass_height, high_pass_height), max(low_pass_width, high_pass_width), color).

    :param low_pass: the array of the low-pass image.
    :type numpy.ndarray

    :param high_pass: the array of the high-pass image.
    :type numpy.ndarray

    :returns returns the two images resized to the same size
    :rtype numpy.ndarray
    """
    # If the images are already the same size, just return
    if low_pass.shape == high_pass.shape:
        return low_pass, high_pass

    # Make new shape with the larger of the two heights and
    # larger of the two widths.
    # The images are assumed to have the same number of colour channels (if at all)
    if low_pass.ndim == high_pass.ndim == 2:
        # Grey-scale
        new_image_shape = (max(low_pass.shape[0], high_pass.shape[0]),
                           max(low_pass.shape[1], high_pass.shape[1]))
    elif low_pass.ndim != high_pass.ndim:
        raise ValueError('Colour channels of low_pass and high_pass must match.')
    else:
        # Colour
        new_image_shape = (max(low_pass.shape[0], high_pass.shape[0]),
                           max(low_pass.shape[1], high_pass.shape[1]),
                           low_pass.shape[2])

    low_height, low_width = low_pass.shape[:2]
    high_height, high_width = high_pass.shape[:2]

    # Calculate positions to centre images within larger, padded container.
    y_start_low = (new_image_shape[0] - low_height) // 2
    x_start_low = (new_image_shape[1] - low_width) // 2

    y_start_high = (new_image_shape[0] - high_height) // 2
    x_start_high = (new_image_shape[1] - high_width) // 2

    # Create larger, padded containers.
    reshaped_low = np.zeros(new_image_shape, dtype=low_pass.dtype)
    reshaped_high = np.zeros(new_image_shape, dtype=high_pass.dtype)

    # Centre the images in the new container arrays.
    reshaped_low[y_start_low:y_start_low + low_height, x_start_low:x_start_low + low_width, :] = low_pass
    reshaped_high[y_start_high:y_start_high + high_height, x_start_high:x_start_high + high_width, :] = high_pass

    return reshaped_low, reshaped_high