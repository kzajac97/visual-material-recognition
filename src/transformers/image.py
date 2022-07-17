from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import rank
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import disk


class ImageTransformerBase:
    """Base class for image transforms. It processes an image returning changed image"""

    @abstractmethod
    def transform(self, x: NDArray) -> NDArray:
        raise NotImplementedError("Can't call base class methods")

    def transform_batch(self, x: NDArray) -> NDArray:
        """Runs transformation function for all example in batch of images"""
        return np.asarray([self.transform(single_image) for single_image in x])


class ColorToGreyTransformer(ImageTransformerBase):
    """Simple transformer converting image to greyscale"""

    def __init__(self):
        super().__init__()

    def transform(self, x: NDArray) -> NDArray:
        """Converts image to grey scale"""
        return rgb2gray(x)


class OtsuThresholdTransformer(ImageTransformerBase):
    """Transformer using Otsu threshold method to convert image to binary"""

    def transform(self, x: NDArray) -> NDArray:
        """Converts image to binary using otsu threshold"""
        threshold = threshold_otsu(x)
        return (x >= threshold).astype(int)


class LocalOtsuThresholdTransformer(ImageTransformerBase):
    """Transformer using Otsu threshold method to convert image to binary"""

    def __init__(self, radius: int):
        """
        :param radius: Radius of local neighbourhood to use for local thresholding in pixels
        """
        self.radius = radius
        super(LocalOtsuThresholdTransformer, self).__init__()

    def transform(self, x: NDArray) -> NDArray:
        """Converts image to binary using local otsu threshold"""
        footprint = disk(self.radius)
        local_thresholds = rank.otsu(x, footprint)
        return (x >= local_thresholds).astype(int)


class CannyTransformer(ImageTransformerBase):
    """Transformer using Canny edge detector"""

    def __init__(self, sigma: float):
        self.sigma = sigma
        super().__init__()

    def transform(self, x: NDArray) -> NDArray:
        return canny(x, sigma=self.sigma)
