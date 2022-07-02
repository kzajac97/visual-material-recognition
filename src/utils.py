from typing import Sequence

import numpy as np


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
