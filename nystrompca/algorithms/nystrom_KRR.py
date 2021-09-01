
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
Kernel ridge regression with the Nyström method

"""

import numpy as np

from nystrompca import KernelRR
from nystrompca.base import NystromMethod
from nystrompca.utils import get_inverse


class NystromKRR(KernelRR, NystromMethod):

    """
    Implements Nyström kernel ridge regression.

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)


    def fit(self, X: np.ndarray,
                  y: np.ndarray) -> None:
        """
        Please see parent class 'nystrompca.base.regression.Regression'
        for documentation

        """
        self._setup(X, y)

        if self.subset is None:
            self.create_subset(self.n)

        K_nm = self.kernel.matrix(self.X, self.X[self.subset])
        K_mm = K_nm[self.subset]

        M = K_nm.T @ K_nm + self.gamma * K_mm
        self.beta = get_inverse(M) @ K_nm.T @ self.y_bar


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

        X_transformed = self.scaler.transform(X_new)

        kappa = self.kernel.matrix(self.X[self.subset], X_transformed)
        predictions = self.alpha + kappa.T @ self.beta

        return predictions

