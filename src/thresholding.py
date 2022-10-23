import numpy as np
from numpy.typing import NDArray
from skimage.filters import rank
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import disk


def otsu_threshold(image: NDArray) -> NDArray:
    """"Converts image to binary using otsu threshold"""
    threshold = threshold_otsu(image)
    return (image >= threshold).astype(int)


def local_threshold_otsu(image: NDArray, radius: int) -> NDArray:
    """Converts image to binary using local otsu threshold"""
    footprint = disk(radius)
    local_thresholds = rank.otsu(image, footprint)
    return (image >= local_thresholds).astype(int)


def count_pixel_values(image_batch: NDArray, normalize: bool = True) -> NDArray:
    """Computes the number of dark and bright pixels in binary image"""
    def count(image):
        n_bright = np.count_nonzero(image)
        n_dark = image.size - n_bright

        if normalize:
            n_dark = n_dark / image.size
            n_bright = n_bright / image.size

        return np.array([n_bright, n_dark])

    return np.asarray([count(image) for image in image_batch])
