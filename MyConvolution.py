import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders

    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """

    # Kernel is inverted before convolution
    kernel = kernel[::-1]

    # Check kernel is suitable
    if kernel.ndim != 2: raise AttributeError('kernel must be of shape (kheight,kwidth)')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1: raise AttributeError('kernel dimensions must both be odd')

    # Check image is suitable
    greyscale = False
    if image.ndim == 2: greyscale = True
    elif image.ndim != 3: raise AttributeError('image must be of shape (rows, cols) or (rows,cols,channels)')


    if greyscale:
        # Only one channel, simply convolve the image
        return convolve_2d(image, kernel)
    else:
        output = np.ndarray(image.shape)
        # Loop through every colour channel in the image,
        # do the convolution on each individual channel, then combine
        for channel in range(image.shape[2]):
            output[:, :, channel] = convolve_2d(image[:, :, channel], kernel)
        return output

def convolve_2d(image_2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a greyscale image with a kernel. Adds zero-padding to the image to handle the borders.

    :param image_2d: the image (greyscale, shape=(rows,cols))
    :type numpy.ndarray

    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray

    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    y, x = image_2d.shape
    kheight, kwidth = kernel.shape

    # Add padding so the convolved image is the same size as the input image.
    # e.g., for a 5x3 kernel, 2 pixels of padding are added to all sides.
    padding = max(int(kheight / 2), int(kwidth / 2))
    padded_image = np.zeros((y + 2 * padding, x + 2 * padding))
    padded_image[padding:-padding, padding:-padding] = image_2d
    output = np.zeros((y, x))

    # Do the convolution
    for i in range(y):
        for j in range(x):
            output[i, j] = np.sum(padded_image[i:i + kheight, j:j + kwidth] * kernel)
    return output