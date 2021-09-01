
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
Calculation of the confidence bound for the
empirical reconstruction error

"""

import numpy as np
from scipy.stats import norm

from nystrompca.base import Kernel
from nystrompca.utils import get_eigendecomposition


def calculate_bound(X:      np.ndarray,
                    n:      int,
                    kernel: str,
                    alpha:  float     = 0.9,
                    **kwargs               ) -> np.ndarray:
    """
    Convenience function to calculate the confidence bound
    from a subset dataset and a kernel name.

    Parameters
    ----------
    X : numpy.ndarray, 2d
        A subset of data points
    kernel : str
        Kernel name
    n : int
        Total size of dataset
    alpha: float
        Confidence level
    **kwargs
        Kernel parameters (please see `nystrompca.base.Kernel`)

    Returns
    -------
    bounds : numpy.ndarray, 1d
        Confidence bound for all PCA dimensions

    """
    kernel_obj = Kernel(kernel=kernel, **kwargs)

    K_mm = kernel_obj.matrix(X, demean=False)

    bounds = calc_conf_bound(K_mm, n, kernel_obj.get_bound(), alpha=alpha)

    return bounds


def calc_conf_bound(K_mm : np.ndarray,
                    n:     int,
                    B:     float,
                    alpha: float     = 0.9) -> np.ndarray:
    """
    Calculation of confidence bound.

    Calculate a high-probability confidence bound on the empirical
    reconstruction error for kernel PCA with the Nyström method for
    all PCA dimensions.

    Parameters
    ----------
    K_mm : numpy.ndarray, 2d
        Subset kernel matrix
    n : int
        Total size of dataset
    B: float
        maximum value of kernel function
    alpha: float
        confidence level

    Returns
    -------
    bounds : numpy.ndarray, 1d
        Confidence bound for all PCA dimensions

    """
    m = K_mm.shape[0]

    if m == n: # Zero error for the Nyström method
        return np.zeros(m)

    L_m, _ = get_eigendecomposition(K_mm / m)

    eig_diff = L_m[:-1] - L_m[1:]

    delta = np.log(2 / (1 - alpha))

    term1   = B * np.sqrt(2 * delta) / np.sqrt(n-m)
    term2_1 = np.sqrt(2 * np.log(2))
    term2_2 = 2 * np.sqrt(2 * np.pi) * norm.cdf(-np.sqrt(2*np.log(2)))
    term2   = B**2 / np.sqrt(m) * (term2_1 + term2_2)
    D = (n - m) / n * (term1 + term2)

    D_k = np.ones(len(eig_diff))
    non_zero = eig_diff > 1e-14
    D_k[non_zero] = D**2 / eig_diff[non_zero]**2
    D_k[D_k > 1] = 1

    max_D = np.array([np.max(D_k[:i+1]) for i in range(m-1)])
    bounds = np.cumsum(L_m[:-1] * D_k) + D * max_D

    # Add NaN for the last PCA dimension
    bounds = np.r_[bounds, np.nan]

    return bounds

