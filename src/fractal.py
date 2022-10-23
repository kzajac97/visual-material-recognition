import numpy as np
from numpy.typing import NDArray


def box_count(x: NDArray, size: int) -> int:
    """Counts th amount of boxes with given size enclosing the shape"""
    enclosed = np.add.reduceat(x, np.arange(0, x.shape[0], size), axis=0)
    enclosed = np.add.reduceat(enclosed, np.arange(0, x.shape[1], size), axis=1)
    return len(np.where((enclosed > 0) & (enclosed < size**2))[0])


def shorter_side_length(x: NDArray) -> int:
    """Returns the length of shorter size of 2D image"""
    return min(x.shape)


def max_power(size: int) -> int:
    """
    Returns greatest power of 2 smaller than given value
    It is used to generate spacing for box sizes
    example:
    >>> self.max_power(45)
    ... 32
    """
    return int(2 ** np.floor(np.log(size) / np.log(2)))


def get_exponent(n: int) -> int:
    """
    Extracts exponent of 2 from value
    example:
        >>> self.get_exponent(16)
        ... 4
    """
    return int(np.log(n) / np.log(2))


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


def fit_value(sizes: NDArray, counts: NDArray) -> float:
    """
    This method computes the fractal dimension of given image
    :param sizes: array of sizes used in box count
    :param counts: value of box count
    :return: fractal dimension as approximate slope of the dependency of sizes and box counts
    """
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -1 * coeffs[0]


def compute_fractal_dimension(x: NDArray) -> np.array:
    """Runs fractal dimension quantifier on binary image"""
    length = shorter_side_length(x)
    power = max_power(length)
    exponent = get_exponent(power)
    box_sizes = power_of_two_space(exponent)
    counts = np.array([box_count(x, size) for size in box_sizes])
    return np.array([fit_value(box_sizes, counts)])
