import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class MeanBaseline(RegressorMixin, BaseEstimator):
    """Baseline estimator always returning mean value of training target"""

    def __init__(self):
        super().__init__()

        self.mean_value = np.nan

    def fit(self, x: NDArray, y: NDArray) -> None:
        """Stores the mean value of target array"""
        self.mean_value = y.mean()

    def predict(self, x: NDArray) -> NDArray:
        """Returns training mean value for each example"""
        return np.full(len(x), fill_value=self.mean_value)

    def fit_predict(self, x: NDArray, y: NDArray) -> NDArray:
        """Calls fit and predict on the same input array"""
        self.fit(x, y)
        return self.predict(x)


class MedianBaseline(RegressorMixin, BaseEstimator):
    """Baseline estimator always returning mean value of training target"""

    def __init__(self):
        super().__init__()

        self.median_value = np.nan

    def fit(self, x: NDArray, y: NDArray) -> None:
        """Stores the median value of target array"""
        self.median_value = np.median(y)

    def predict(self, x: NDArray) -> NDArray:
        """Returns training median value for each example"""
        return np.full(len(x), fill_value=self.median_value)

    def fit_predict(self, x: NDArray, y: NDArray) -> NDArray:
        """Calls fit and predict on the same input array"""
        self.fit(x, y)
        return self.predict(x)
