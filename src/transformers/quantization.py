import numpy as np
from numpy.typing import NDArray

from src.transformers.image import ImageTransformerBase

"""
QuantizationTransformers are ImageTransformers which convert image into numerical features
They use the same interface as ImageTransformers, but return smaller sized arrays with rank 1
"""


class BinaryQuantizationTransformer(ImageTransformerBase):
    """
    Quantization Transformer processing thresholded images
    It computes the number of dark and bright pixels in binary image
    """

    def __init__(self, normalize: bool = True):
        """
        :param normalize: if True returned values are dark/bright pixel ratios with respect to image size
        """
        self.normalize = normalize

    @staticmethod
    def _normalize_pixel_count(value: float, image_size: int) -> float:
        """Normalizes pixel count with respect to given image"""
        return value / image_size

    def transform(self, x: NDArray) -> NDArray:
        """Computes the number of dark and bright pixels in binary image"""
        n_bright = np.count_nonzero(x)
        n_dark = x.size - n_bright

        if self.normalize:
            n_dark = self._normalize_pixel_count(n_dark, x.size)
            n_bright = self._normalize_pixel_count(n_bright, x.size)

        return np.array([n_bright, n_dark])


class FractalDimensionQuantizationTransformer(ImageTransformerBase):
    """
    This class implements fractal dimension quantifier
    It is a measure of roughness of the curve or shape on the image

    It is implemented using box count method, assuming the more smaller boxes is required to
    cover some shape and fewer smaller boxes, the higher the fractal dimension of given shape
    """

    @staticmethod
    def box_count(x: NDArray, size: int) -> int:
        """Counts th amount of boxes with given size enclosing the shape"""
        enclosed = np.add.reduceat(x, np.arange(0, x.shape[0], size), axis=0)
        enclosed = np.add.reduceat(enclosed, np.arange(0, x.shape[1], size), axis=1)

        return len(np.where((enclosed > 0) & (enclosed < size**2))[0])

    @staticmethod
    def shorter_side_length(x: NDArray) -> int:
        """Returns the length of shorter size of 2D image"""
        return min(x.shape)

    @staticmethod
    def max_power(size: int) -> int:
        """
        Returns greatest power of 2 smaller than given value
        It is used to generate spacing for box sizes

        example:
        >>> self.max_power(45)
        ... 32
        """
        return int(2 ** np.floor(np.log(size) / np.log(2)))

    @staticmethod
    def exponent(n: int) -> int:
        """
        Extracts exponent of 2 from value

        example:
            >>> self.exponent(16)
            ... 4
        """
        return int(np.log(n) / np.log(2))

    @staticmethod
    def power_of_two_space(max_exponent: int) -> NDArray:
        """
        Return spacing of powers of two up to given max exponent
        Methods starts from 2 ** 2, so 1 and 2 are skipped

        example:
            >>> self.power_of_two_space(5)
            ... array([32, 16, 8, 4])
        """
        exponents = np.arange(max_exponent, 1, -1)
        return np.power(2, exponents)

    @staticmethod
    def fit_value(sizes: NDArray, counts: NDArray) -> float:
        """
        This method computes the fractal dimension of given image

        :param sizes: array of sizes used in box count
        :param counts: value of box count

        :return: fractal dimension as approximate slope of the dependency of sizes and box counts
        """
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -1 * coeffs[0]

    def transform(self, x: NDArray) -> float:
        """Runs fractal dimension quantifier on binary image"""
        length = self.shorter_side_length(x)
        power = self.max_power(length)
        exponent = self.exponent(power)
        box_sizes = self.power_of_two_space(exponent)

        counts = np.array([self.box_count(x, size) for size in box_sizes])
        return self.fit_value(box_sizes, counts)
