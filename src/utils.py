from functools import wraps
from typing import Any, List, Sequence, Callable

import numpy as np
import pandas as pd
from skimage import io


def sliding_window_sum(a: np.array, window_size: int) -> np.array:
    """
    Computes sums over sliding window in given array

    :param a: input array
    :param window_size: size of sliding window

    example:
        >>> sliding_window_sum(np.ones(10, dtype=int), window_size=5)
        ... np.array([5., 5., 5., 5., 5., 5.])
        >>> sliding_window_sum(np.arange(10, dtype=int), window_size=5)
        ... np.array([[10, 15, 20, 25, 30, 35]])

    :return: array with computed sliding window sums
    """
    if window_size > len(a):
        raise ValueError(f"Windows size must be smaller than number of elements! {window_size} > {len(a)}!")
    return np.convolve(a, np.ones(window_size, dtype=int), "valid")


def cast_to_arrays(y_true: Sequence, y_pred: Sequence) -> tuple:
    """Converts any pair of Sequences into numpy arrays"""
    return np.asarray(y_true), np.asarray(y_pred)


def verify_shape(y_true: np.array, y_pred: np.array) -> None:
    """Verifies if arrays have correct shape for metric computation"""
    if y_true.shape != y_pred.shape:
        raise TypeError(f"Array shapes are different must match! {y_true.shape} != {y_pred.shape}")


def sample_one(column: pd.Series) -> Any:
    """Samples one element from pandas Series"""
    return column.sample(1).tolist()[0]


def load_image_batch(paths: List[str]) -> np.array:
    """Loads images from paths into a stacked array"""
    return np.asarray([io.imread(path) for path in paths])


def to_batch_function(func: Callable) -> Callable:
    """Converts any numpy function to work with batches"""
    @wraps(func)
    def wrapper(batch: np.array) -> np.array:
        return np.asarray([func(sample) for sample in batch])

    return wrapper
