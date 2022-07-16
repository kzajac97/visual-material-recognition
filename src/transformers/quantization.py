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
    @staticmethod
    def _validate_input_image(x: NDArray) -> None:
        """Checks shape of the input image, must be 2D"""
        if x.ndim != 2:
            raise ValueError(f"Image must have rank 2! {x.ndim} != 2")

    def transform(self, x: NDArray) -> NDArray:
        """Computes the number of dark and bright pixels in binary image"""
        self._validate_input_image(x)
        n_dark = np.count_nonzero(x)
        n_bright = x.size - n_dark

        return np.array([n_bright, n_dark])


class FractalDimensionQuantizationTransformer(ImageTransformerBase):
    def transform(self, x: NDArray) -> NDArray:
        ...
