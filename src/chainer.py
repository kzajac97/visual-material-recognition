from typing import List

import numpy as np
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

    def transform(self, x: NDArray, batch: bool = True) -> NDArray:
        """
        Runs a chain of transformations specified in __init__

        :param x: batch of images or single image
        :param batch: if True input contains multiple images to process

        :return: processed images or quantity
        """
        for transformer in self.transformers:
            if batch:
                x = transformer.transform_batch(x)
            else:
                x = transformer.transform(x)

        return x


class ParallelTransformChainer:
    def __init__(self, transformers: List[TransformChainer]):
        self.chained_transformers = transformers

    def transform(self, x: NDArray, batch: bool = True) -> NDArray:
        """
        Runs multiple transform chainers in parallel

        :param x: batch of images or single image
        :param batch: if True input contains multiple images to process

        :return: array of processed
        """
        results = []
        for chainer in self.chained_transformers:
            results.append(chainer.transform(x, batch=batch))

        return np.column_stack(results)

    def transform_multiprocessing(self, x: NDArray, batch: bool = True) -> NDArray:
        """
        Runs multiple transform chainers in parallel

        # TODO: Implement actual multiprocessing here

        :param x: batch of images or single image
        :param batch: if True input contains multiple images to process

        :return: array of processed
        """
        results = []
        for chainer in self.chained_transformers:
            results.append(chainer.transform(x, batch=batch))

        return np.column_stack(results)
