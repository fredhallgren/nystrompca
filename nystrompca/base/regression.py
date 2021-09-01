
# Copyright 2021 Fredrik Hallgren
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Abstract base class for regression methods that have dependent and
independent variables and calculates a constant (regression intercept)
plus independent variable coefficients.

"""


from abc import ABC, abstractmethod

import numpy as np

from nystrompca.utils import to_column_vector


class Regression(ABC):

    """
    Abstract base class for regression methods.

    Attributes
    ----------
    alpha : float
        Regression constant
    beta : numpy.ndarray
        Regression feature coefficients

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.X:    np.ndarray = None

        self.y:    np.ndarray = None

        self.n:           int = None

        self.alpha:       int = None

        self.beta: np.ndarray = None


    @abstractmethod
    def fit(self, X: np.ndarray,
                  y: np.ndarray) -> None:
        """
        Fit the regression model.

        Parameters
        ----------
        X : numpy.ndarray, 2d
            Data matrix used for predictions
        y : numpy.ndarray, 1d or 2d
            True targets

        """
        ...


    @abstractmethod
    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Create predictions from the fitted model.

        Parameters
        ----------

        X: numpy.ndarray, 1d or 2d
            New test points to create predictions for. Either 1d or
            a 2d row vector for one data point, or a matrix with multiple
            data points in the rows.

        Returns
        -------
        predictions : numpy.ndarray, 2d
            A column vector of predictions

        """
        ...


    def score(self, X: np.ndarray,
                    y: np.ndarray) -> float:
        """
        Calculate the regression R-squared.

        Parameters
        ----------
        X : numpy.ndarray, 2d
            Data matrix used for predictions
        y : numpy.ndarray, 1d or 2d
            True targets

        Returns
        -------
        R_squared : float
            R-squared

        """
        y = to_column_vector(y)

        predictions = self.predict(X)

        RSS = np.sum((y - predictions)**2)

        TSS = np.sum((y - y.mean())**2)

        R_squared = 1 - RSS / TSS

        return R_squared

