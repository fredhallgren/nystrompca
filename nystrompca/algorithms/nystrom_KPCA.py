
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
Kernel principal components analysis with the Nyström method.

"""

from typing import Tuple

import numpy as np

from nystrompca import KernelPCA
from nystrompca.base import NystromMethod, Kernel
from nystrompca.utils import (get_inverse, get_eigendecomposition,
                              get_kappa, flip_dimensions, demean_matrix)


class NystromKPCA(KernelPCA, NystromMethod):

    """
    Implements Nyström kernel PCA

    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.K_mm_p: np.ndarray = None
        self.K_mm:   np.ndarray = None
        self.K_nm_p: np.ndarray = None
        self.K_nm:   np.ndarray = None


    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the Nyström kernel PCA for the supplied data matrix.

        """
        self._setup(X)

        if self.subset is None:
            self.create_subset(self.n)

        K_mm_p, K_mm, K_nm_p, K_nm = self.get_kernel_matrices()

        K_inv_sqrt = get_inverse(K_mm_p, np.sqrt)

        nystrom_matrix = K_inv_sqrt @ K_nm_p.T @ K_nm_p @ K_inv_sqrt / self.n
        M, V = get_eigendecomposition(nystrom_matrix)

        components_ = K_inv_sqrt @ V[:,:self.n_components]
        scores_     = K_nm_p @ components_

        scores_, components_ = flip_dimensions(scores_, components_)

        self.all_variances       = M.copy()
        self.explained_variance_ = M[:self.n_components]
        self.components_         = components_
        self.scores_             = scores_
        self.K_mm_p              = K_mm_p
        self.K_mm                = K_mm
        self.K_nm_p              = K_nm_p
        self.K_nm                = K_nm

        return self.scores_


    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Transform data into the coordinate system defined by the
        approximate Nyström principal components

        Raises
        ------
        ValueError
            If the 'fit_transform' method has not been called yet

        """
        if self.components_ is None:
            raise ValueError("Call 'fit_transform' before this function.")

        X_new = self.scaler.transform(X_new)

        kappa = self.kernel.matrix(self.X[self.subset], X_new, demean=False)

        kappa_p = self.get_kappa_p(kappa)

        X_transformed = kappa_p.T @ self.components_

        return X_transformed


    def get_reconstruction_errors(self, approx: bool = False) -> np.ndarray:
        """
        Calculate the reconstruction errors of the dataset onto
        the Nyström principal components.

        This method requires contruction of the full kernel matrix,
        be mindful that this is compute intensive. There is an option
        to create an approximate error which takes O(nm^2).

        Parameters
        ----------
        approx : bool
            Whether to approximate the mean of the full kernel matrix.

        Returns
        -------
        errors_ : numpy.ndarray
            Reconstruction errors

        Raises
        ------
        ValueError
            If the 'fit_transform' method has not been called yet

        """
        n = self.n
        X = self.X

        if self.components_ is None:
            raise ValueError("Call 'fit_transform' before this function.")

        if not self.demean:
            tot_var = np.sum([self.kernel(X[i], X[i]) for i in range(n)]) / n

        elif not approx:
            if self.K_p is None:
                self.K_p = self.kernel.matrix(X, demean=True)
            tot_var = np.trace(self.K_p) / n

        else:
            K_trace = np.array([self.kernel(X[i], X[i]) for i in range(n)])
            row_means = self.K_nm.sum(1) / len(self.subset)
            full_mean = row_means.sum() / n
            tot_var = np.sum(K_trace - 2 * row_means + full_mean) / n

        self.errors_ = tot_var - np.cumsum(self.explained_variance_)

        return self.errors_


    def get_subset_errors(self) -> np.ndarray:
        """
        Calculate the reconstruction error of all n data points onto
        the eigenspace from the subset of m subsampled points

        Returns
        -------
        errors : numpy.ndarray, 1d
            Reconstruction errors for all PCA dimensions

        Raises
        ------
        ValueError
            If the 'fit_transform' method has not been called yet

        """
        if self.components_ is None:
            raise ValueError("Call 'fit_transform' before this function.")

        if self.K_p is None:
            self.K_p = self.kernel.matrix(self.X, self.X, demean=self.demean)

        L_m, U_m = get_eigendecomposition(self.K_mm_p)

        j = np.where(L_m > 0)[0][-1]

        projections = np.zeros(self.n_components)

        for i in range(self.n_components):

            k = min(i,j)
            L_k = np.diag(1 / L_m[:k+1])
            U_k = U_m[:,:k+1]
            mat = self.K_nm_p @ U_k @ L_k @ U_k.T @ self.K_nm_p.T
            projections[i] = np.trace(mat)

        errors = np.trace(self.K_p) / self.n - projections / self.n

        return errors


    def get_subset_variance(self, X_new:  np.ndarray,
                                  PC_idx: int       ) -> float:
        """
        Calculate the amount of variance from new data captured
        by a Subset PCA principal components.

        Parameters
        ----------
        X_new : numpy.ndarray, 2d
            Data variance to capture
        PC_idx : int
            Which PC to consider

        Returns
        -------
        subset_variance: float
            Variance captured by one principal component

        Raises
        ------
        ValueError
            If 'fit_transform' has not been called yet

        """
        if self.K_mm_p is None:
            raise ValueError("Call 'fit_transform' before using this method.")

        kappa = self.kernel.matrix(self.X[self.subset], X_new, demean=False)
        kappa_p = self.get_kappa_p(kappa)

        L, U = get_eigendecomposition(self.K_mm_p)
        M = U[:,PC_idx].T @ kappa_p @ kappa_p.T @ U[:,PC_idx]
        eigenvalue_norm = X_new.shape[0] * L[PC_idx]
        if np.abs(eigenvalue_norm) < 1e-12:
            subset_variance = 0
        else:
            subset_variance = M / eigenvalue_norm

        return subset_variance


    def get_kernel_matrices(self) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray,
                                           np.ndarray]:
        """
        Calculate the kernel matrices K_mm and K_nm

        Parameters
        ----------
        X: numpy.ndarray, 2d
            Data matrix
        subset: numpy.ndarray
            Subset indices
        kernel: nystrom_KPCA.kernel.Kernel
            Kernel wrapper class

        Returns
        -------
        K_mm_p : numpy.ndarray, 2d
            Centred kernel matrix with size m x m
        K_nm_p : numpy.ndarray, 2d
            Centred kernel matrix with size n x m
        K_nm : numpy.ndarray, 2d
            Kernel matrix with size n x m

        """
        K_nm = self.kernel.matrix(self.X, self.X[self.subset], demean=False)

        K_mm = K_nm[self.subset]

        if self.demean:
            K_mm_p, K_nm_p = demean_matrices(K_mm, K_nm)

        else:
            K_mm_p = K_mm
            K_nm_p = K_nm

        return K_mm_p, K_mm, K_nm_p, K_nm


    def get_kappa_p(self, kappa):
        """
        Get kappa as used in the nystrom demeaning

        """
        if not self.demean:
            kappa_p = kappa

        else:
            n, m = self.K_nm.shape
            n_new = kappa.shape[1]

            n_mean = self.K_nm.sum(0) / n

            M1 = np.tile(n_mean[:,np.newaxis], (1, n_new))
            m0 = n_mean @ get_inverse(self.K_mm)
            M2 = np.tile(m0 @ kappa, (m,1))
            M3 = n_mean @ m0

            kappa_p = kappa - M1 - M2 + M3

        return kappa_p




#############
### UTILS ###
#############


def demean_matrices(K_mm, K_nm):
    """
    Demean the matrices K_mm and K_nm

    """
    n, m = K_nm.shape

    n_mean = K_nm.sum(0) / n

    M1 = np.tile(n_mean, (n,1))

    m0 = get_inverse(K_mm) @ n_mean[:,np.newaxis]

    M2 = np.tile(K_nm @ m0, (1, m))

    M3 = n_mean @ m0

    K_nm_p = K_nm - M1 - M2 + M3

    M1 = M1[:m]

    K_mm_p = K_mm - M1 - M1.T + M3

    return K_mm_p, K_nm_p
