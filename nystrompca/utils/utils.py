
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
Various utility functions.

"""

from typing import Tuple, Callable

import numpy as np


def demean_matrix(M: np.ndarray) -> np.ndarray:
    """
    Subtract the mean in feature space from a kernel matrix.

    Parameters
    ----------
    M : numpy.ndarray, 2d
        Matrix to be mean-adjusted

    Returns
    -------
    M_p : numpy.ndarray
        Adjusted matrix

    """
    m, n = M.shape

    M1 = M.sum(0).repeat(m).reshape(n,m).T / m
    M2 = M.sum(1).repeat(n).reshape(m,n) / n
    M3 = np.sum(M) / (m*n)
    M_p = M - M1 - M2 + M3

    return M_p


def get_eigendecomposition(M:   np.ndarray,
                           eps: float     = 1e-12
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the eigendecomposition of a symmetric positive semi-definite
    matrix. Sets negative eigenvalues or eigenvalues smaller than "eps"
    to zero.

    Parameters
    ----------
    M: numpy.ndarray, 2d
        Symmetric positive semi-definite matrix

    Returns
    -------
    L : numpy.ndarray, 1d
        Eigenvalues, ordered in decreasing order
    U : np.ndarray, 2d
        Eigenvectors

    """
    L, U = np.linalg.eigh(M)
    L = L[::-1]
    U = U[:,::-1]

    # Sometimes the matrix can have small negative eigenvalues due to numerical
    # inaccuracies, despite being positive semidefinite in theory. Set these
    # to zero. Also remove tiny eigenvalues for numerical regularization.
    L[L < eps] = 0

    return L, U


def get_inverse(M:    np.ndarray,
                func: Callable  = lambda x: x,
                eps:  float     = 1e-9) -> np.ndarray:
    """
    Calculate the regularized inverse, removing small or negative eigenvalues
    before taking the inverse.

    Parameters
    ----------
    M : numpy.ndarray, 2d square
        matrix to be inverted
    func : Callable, default identity
        Transformation to apply to the eigenvalues before
        taking the inverse

    Returns
    -------
    M_inv : numpy.ndarray, square 2d
        Inverse matrix

    """
    L, U = get_eigendecomposition(M, eps)

    j = np.where(L > 0)[0][-1]
    M_inv = U[:,:j+1] @ np.diag(1 / func(L[:j+1])) @ U[:,:j+1].T

    return M_inv


def get_tail_sums(L: np.ndarray,
                  d: int       = None) -> np.ndarray:
    """
    Calculate the first d sums of the tail of an array.

    Parameters
    ----------
    L : numpy.ndarray, 1d
        Numbers to be summed
    d : int, optional
        Number of tail sums to compute. If None then compute all
        tail sums.

    Returns
    -------
    L_sums : np.ndarray
        Tail sums

    """
    if d is None:
        d = len(L)

    L_sums = (L.sum() - L.cumsum())[:d]

    return L_sums


def get_kappa(kappa:  np.ndarray,
              K:      np.ndarray,
              demean: bool      = True) -> np.ndarray:
    """
    Demean the kappa vector of kernel evaluations of new data points
    with the the existing ones. Works either for one new data point, in
    which case "kappa" should be a column vector, or for multiple data
    points with "kappa" a matrix.

    Parameters
    ----------
    kappa : numpy.ndarray, 2d
        Kernel evaluations of new data points
    K : numpy.ndarray, 2d
        Uncentred kernel matrix, with the same data points used in "kappa"

    Returns
    -------
    kappa_p
        Centred kernel evaluations

    """
    if not demean:
        kappa_p = kappa

    else:
        m, n = K.shape
        n_new = kappa.shape[1]

        kappa_mean = np.tile(kappa.sum(0), (m, 1)) / m

        n_mean = np.tile(K.sum(1)[:, np.newaxis], n_new) / n

        nm_mean = np.tile(K.sum() / (n * m), (m, n_new))

        kappa_p = kappa - kappa_mean - n_mean + nm_mean

    return kappa_p


def to_column_vector(y: np.ndarray) -> np.ndarray:
    """
    Take a one-dimensional vector or a row vector and
    convert into a column vector.

    Parameters
    ----------
    y : numpy.ndarray, 1d or 2d
        Vector to convert

    Returns
    -------
    y : numpy.ndarray, 2d
        Column vector

    """
    if y.ndim == 1:
        y = y[:, np.newaxis]

    if y.ndim == 2 and y.shape[0] == 1:
        y = y.T

    return y


def flip_dimensions(scores:     np.ndarray,
                    components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    The principal components and scores are only unique up to a sign,
    so we flip the dimensions so that the range of scores in each dimenstion
    lies mostly within positive values. Specifically, we flip dimensions
    so that the mid point of the range of values is positive.

    Parameters
    ----------
    scores : numpy.ndarray, 2d
        Use these to calculate which dimensions to flip and flip them
    components : numpy.ndarray, 2d
        Flip these too

    Returns
    -------
    scores_flipped : numpy.ndarray, 2d
        Flipped vectors
    components_flipped : numpy.ndarray, 2d
        Flipped vectors

    """
    flip = (scores.min(0) + scores.max(0)) / 2 < 0
    flip_matrix = np.diag(1 - 2 * flip)

    scores_flipped     = scores     @ flip_matrix
    components_flipped = components @ flip_matrix

    return scores_flipped, components_flipped


class IdentityScaler:
    """
    This class is used to scale the input data points
    when no adjustment is desired and the input data is
    used as is.

    This class is used in the unit tests.

    """

    @classmethod
    def fit_transform(cls, X: np.ndarray) -> np.ndarray:
        """
        Do nothing

        """
        return X

    @classmethod
    def transform(cls, X: np.ndarray) -> np.ndarray:
        """
        Do nothing

        """
        return X

