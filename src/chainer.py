from typing import List

from numpy.typing import NDArray

from src.transformers.image import ImageTransformerBase


class TransformChainer:
    """
    Transformer chaining multiple transformations together

    The Example shows how to create feature extractor running following steps:
        1. Conversion to greyscale
        2. Otsu thresholding
        3. Binary quantization of dark and bright pixels

    example:
        >>> chainer = TransformChainer(transformers=[
        ...     ColorToGreyTransformer(),
        ...     OtsuThresholdTransformer(),
        ...     BinaryQuantizationTransformer()
        ... ])
    """
    def __init__(self, transformers: List[ImageTransformerBase]):
        self.transformers = transformers

    def transform(self, x: NDArray) -> NDArray:
        """Runs all specified transformations returning the output of the last one"""
        for transformer in self.transformers:
            x = transformer.transform(x)

        return x
