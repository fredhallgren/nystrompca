
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
Kernel ridge regression

Penalized non-parametric regression through L2-regularized kernel
regression(aka ridge/Tykhonov regularization). It's often important
to regularize kernel regression due to the large number of parameters.

"""


import numpy as np
from sklearn.preprocessing import StandardScaler

from nystrompca.base import KernelMachine, Regression
from nystrompca.utils import to_column_vector, get_inverse


class KernelRR(KernelMachine, Regression):

    """
    Kernel ridge regression

    Parameters
    ----------
    gamma : float
        Ridge parameter

    """
    def __init__(self, gamma: float = 1e-14, **kwargs):

        super().__init__(**kwargs)

        self.gamma = gamma

        self.scaler = StandardScaler()


    def _setup(self, X: np.ndarray,
                  y: np.ndarray) -> None:
        """
        Preparations to fit the method

        Including required intial transformations of the X and y
        variables. Used by this method and Nyström KRR.

        """
        self.n = X.shape[0]

        self.X = self.scaler.fit_transform(X)

        y = to_column_vector(y)

        self.y_bar = y - np.mean(y)

        self.alpha = np.mean(y) # type: ignore[assignment]


    def fit(self, X: np.ndarray,
                  y: np.ndarray) -> None:
        """
        Please see parent class 'nystrompca.base.regression.Regression'
        for documentation

        """
        self._setup(X, y)

        self.K = self.kernel.matrix(self.X)

        M = self.gamma + self.K
        self.beta = get_inverse(M) @ self.y_bar


    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Please see parent class 'nystrompca.base.regression.Regression'
        for documentation

        Raises
        ------
        ValueError
            If the 'fit' method has not been called yet

        """
        if self.beta is None:
            raise ValueError("Call 'fit' before this function.")

        X_new = self.scaler.transform(X_new)

        kappa = self.kernel.matrix(self.X, X_new)
        predictions = self.alpha + kappa.T @ self.beta

        return predictions
