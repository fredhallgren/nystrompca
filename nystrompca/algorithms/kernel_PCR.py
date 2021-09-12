
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
Kernel principal components regression.

"""

import numpy as np

from nystrompca import KernelPCA
from nystrompca.base import Regression
from nystrompca.utils import to_column_vector


class KernelPCR(KernelPCA, Regression):

    """
    Implements kernel PCR

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)


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
        y = to_column_vector(y)

        y_bar = np.mean(y)

        self.fit_transform(X)

        j = np.where(self.explained_variance_ > 0)[0][-1]
        Lambda_inv = np.diag(1 / (self.n * self.explained_variance_[:j+1]))

        self.beta = np.sqrt(Lambda_inv) @ self.Q[:,:j+1].T @ (y - y_bar)

        self.beta = np.r_[self.beta, np.zeros((self.n_components-(j+1),1))]

        self.alpha = y_bar # type: ignore[assignment]


    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Create predictions from the fitted model.

        Parameters
        ----------
        X_new: numpy.ndarray, 1d or 2d
            New test points to create predictions for. Either 1d or
            a 2d row vector for one data point, or a matrix with multiple
            data points in the rows.

        Returns
        -------
        predictions : numpy.ndarray, 2d
            A column vector of predictions

        Raises
        ------
        ValueError
            If the 'fit' method has not been called yet

        """
        if self.beta is None:
            raise ValueError("Call 'fit' before this function.")

        X_transformed = self.transform(X_new)

        predictions = self.alpha + X_transformed @ self.beta

        return predictions


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

