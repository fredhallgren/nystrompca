
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
Kernel principal component regression with the Nyström method.

"""

import numpy as np

from nystrompca import NystromKPCA
from nystrompca.base import Regression
from nystrompca.utils import to_column_vector


class NystromKPCR(NystromKPCA, Regression):
    """
    Implements Nyström kernel PCR

    """

    @staticmethod
    def from_pca(pca: NystromKPCA):
        """
        Create a Nyström PCR instance using the configuration
        from a Nyström PCA instance.

        """
        nystrom_kpcr = NystromKPCR(n_components = pca.n_components,
                                   m_subset     = pca.m,
                                   kernel       = pca.kernel.name,
                                   sigma        = pca.kernel.sigma,
                                   degree       = pca.kernel.degree,
                                   coef0        = pca.kernel.coef0,
                                   normalize    = pca.kernel.normalize)

        return nystrom_kpcr


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

        self.fit_transform(X)

        self.j = np.where(self.explained_variance_ > 0)[0][-1]
        M_inv = np.diag(1 / (self.explained_variance_[:self.j+1] * self.n))

        self.beta = M_inv @ self.components_.T[:self.j+1] @ self.K_nm_p.T @ y

        self.alpha = np.mean(y) # type: ignore[assignment]


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

        X_transformed = self.transform(X_new)

        predictions = self.alpha + X_transformed[:,:self.j+1] @ self.beta

        return predictions

