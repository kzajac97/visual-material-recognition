from abc import ABC, abstractmethod

from numpy.typing import NDArray
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters.thresholding import threshold_otsu


class ImageTransformerBase(ABC):
    """Base class for image transforms. It processes an image returning changed image"""

    @abstractmethod
    def transform(self, x: NDArray) -> NDArray:
        raise NotImplementedError("Can't call base class methods")


class ColorToGreyTransformer(ImageTransformerBase):
    """Simple transformer converting image to greyscale"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_input_image(x: NDArray) -> None:
        """Checks shape of the input image, must be 3D"""
        if x.ndim != 3:
            raise ValueError(f"Image must have rank 3! {x.ndim} != 3")

    def transform(self, x: NDArray) -> NDArray:
        """Converts image to grey scale"""
        self._validate_input_image(x)
        return rgb2gray(x)


class OtsuThresholdTransformer(ImageTransformerBase):
    """Transformer using Otsu threshold method to convert image to binary"""

    def __init__(self):
        super(OtsuThresholdTransformer, self).__init__()

    @staticmethod
    def _validate_input_image(x: NDArray) -> None:
        """Checks shape of the input image, must be 2D"""
        if x.ndim != 2:
            raise ValueError(f"Image must have rank 2! {x.ndim} != 2")

    def transform(self, x: NDArray) -> NDArray:
        """Converts image to binary using otsu threshold"""
        self._validate_input_image(x)
        threshold = threshold_otsu(x)
        return x > threshold


class CannyTransformer(ImageTransformerBase):
    """Transformer using Canny edge detector"""

    def __init__(self, sigma: float):
        self.sigma = sigma
        super(CannyTransformer, self).__init__()

    def transform(self, x: NDArray) -> NDArray:
        return canny(x, sigma=self.sigma)
