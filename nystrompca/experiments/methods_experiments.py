
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
Experiments comparing Nyström kernel PCA with other
unsupervised learning methods

* Linear PCA
* Kernel PCA
* Sparse PCA
* Locally Linear Embedding
* Independent Component Analysis

As a measure of the accuracy we use the average reconstruction errors
of a hold-out data set by default.

"""

# Built-in modules
import os
import sys
import time
import argparse
import warnings
from typing import Tuple

# Third-party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, SparsePCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding

# This package
from nystrompca import KernelPCA, NystromKPCA
from nystrompca.experiments import base_parser, data
from nystrompca.utils import logger


# Always display DataFrame numbers in decimal format, using 4 decimals
pd.options.display.float_format = '{:,.4f}'.format


def main(kernel:   str  = 'rbf',
         n:        int  = 1000,
         m:        int  = 100,
         d:        int  = 10,
         ica:      bool = False,
         insample: bool = False,
         seed:     int  = None,
         noplot:   bool = False) -> int:
    """
    Run the experiments.

    Calculates the reconstruction error

    Parameters
    ----------
    kernel : str
        Kernel function to use
    n : int
        Total size of dataset
    m : int
        Size of subsampled dataset
    d : int
        Maximum subspace dimension
    ica : bool, default False
        If True then run ICA, which can be time-consuming
    insample : bool, default False
        If True then train and test the methods on the same data
    seed : int, default None
        Random seed for Nyström subset and train/test split
    noplot : bool, default False
        If True then don't display plots

    Returns
    -------
    int
        Return code, 0 if success

    """
    if noplot:
        os.environ['NYSTROMPCA_NOPLOT'] = "NOPLOT"

    logger.header("Methods comparison experiments")

    logger.info(f"Kernel: {kernel}")
    logger.info(f"n: {n}")
    logger.info(f"m: {m}")

    # Turn warnings into errors
    warnings.simplefilter('error')

    all_results = pd.DataFrame()

    t_nys_tot  = 0
    t_kpca_tot = 0

    for dataset in ('magic', 'yeast', 'cardiotocography', 'segmentation',
                    'drug', 'digits', 'dailykos', 'nips'):

        logger.subheader(f"Dataset: {dataset}")

        X = getattr(data, "get_" + dataset + "_data")(n)

        results, t_nys, t_kpca = run_one_dataset(dataset, kernel, X, m, d,
                                                 ica, insample, seed)
        t_nys_tot  += t_nys
        t_kpca_tot += t_kpca

        logger.info("Results:")
        logger.info(results)

        results['dataset'] = dataset

        all_results = all_results.append(results)

    logger.info(f"Kernel PCA elapsed time:  {t_kpca_tot / 8:.3f}")
    logger.info(f"Nyström PCA elapsed time: {t_nys_tot  / 8:.3f}")

    return 0


def run_one_dataset(dataset:  str,
                    kernel:   str,
                    X:        np.ndarray,
                    m:        int,
                    d:        int,
                    ica:      bool,
                    insample: bool,
                    seed:     int      ) -> pd.DataFrame:
    """
    Run an experiment for one kernel and dataset for all methods.

    Parameters
    ----------
    dataset : str
        The name of the dataset
    kernel : str
        Kernel function name
    X : numpy.ndarray, 2d
        Data matrix, with observations in the rows
    m : int
        The size of the Nyström subset
    d : int
        Subspace dimension
    ica : bool, default False
        If True then run ICA
    insample : bool, default False
        If True then don't use a test set
    seed : int, default None
        Random seed

    Returns
    -------
    results : pandas.DataFrame
        The results of one experiment

    """
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=seed)

    if insample:
        X_train, X_test = X, X

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = pd.DataFrame()

    (results['Subset PCA'],
     results['Nystrom PCA'],
     t_nys,
     sigma)                  = run_nystrom_kpca(X_train, X_test, dataset,
                                                kernel, m, d, seed)

    (results['Kernel PCA'],
     t_kpca)                 = run_kernel_pca(X_train, X_test,
                                              kernel, d, sigma)

    results['Linear PCA']    = run_linear_pca(X_train, X_test, d)

    results['Sparse PCA']    = run_sparse_pca(X_train, X_test, d)

    results['LLE']           = run_lle(X_train, X_test, d)

    if ica:
        results['ICA']       = run_ica(X_train, X_test, d)

    results['d'] = range(1,d+1)

    return results, t_nys, t_kpca


def run_nystrom_kpca(X_train: np.ndarray,
                     X_test:  np.ndarray,
                     dataset: str,
                     kernel:  str,
                     m:       int,
                     d:       int,
                     seed:    int       ) -> Tuple[np.ndarray,
                                                   np.ndarray,
                                                   float,
                                                   float]:
    """
    Run Subset PCA and Nyström Kernel PCA experiment.

    Using the same subset for both methods for comparability.

    Parameters
    ----------
    X_train : numpy.ndarray, 2d
        Training data
    X_test : numpy.ndarray, 2d
        Validation data set, with the same number of data dimensions
        as the training set
    dataset: str
        The name of the dataset
    kernel: str
        Kernel function name
    m : int
        Size of Nyström subset
    d : int
        Maximum number of components
    seed : int
        Random seed

    Returns
    -------
    fractions_subset : numpy.ndarray, 1d
        The fraction of explained variance for each dimension for Subset PCA
    fractions : numpy.ndarray, 1d
        The fraction of explained variance for each dimension for Nyström PCA
    t_nys : float
        The time taken to fit the method
    sigma : float
        The kernel bandwidth parameter, to use the same for standard KPCA

    """
    variances = np.zeros(d)

    sigma = 500
    
    nystrom_kpca = NystromKPCA(kernel=kernel, sigma=sigma, n_components=d,
                               m_subset=m, seed=seed)

    # Calculate bandwidth parameter based on the median heuristic
    if dataset not in ('dailykos', 'nips'):
        nystrom_kpca.create_subset(X_train.shape[0])
        nystrom_kpca.kernel.calc_sigma(X_train[nystrom_kpca.subset])
        sigma = nystrom_kpca.kernel.sigma

    t0 = time.process_time()
    nystrom_kpca.fit_transform(X_train)
    t_nys = time.process_time() - t0

    variances_output = nystrom_kpca.transform(X_test).var(0)

    variances[:len(variances_output)] = variances_output # type: ignore[arg-type]

    # Subset PCA errors
    variances_subset = np.zeros(d)
    for i in range(d):
        variances_subset[i] = nystrom_kpca.get_subset_variance(X_test, i)

    # Calculate the fractions of the variance captured
    K = nystrom_kpca.kernel.matrix(X_test)

    total_variance = np.trace(K) / X_test.shape[0]

    fractions = np.cumsum(variances) / total_variance

    fractions_subset = np.cumsum(variances_subset) / total_variance

    return fractions_subset, fractions, t_nys, sigma


def run_kernel_pca(X_train: np.ndarray,
                   X_test:  np.ndarray,
                   kernel:  str,
                   d:       int,
                   sigma:   float     ) -> Tuple[np.ndarray, float]:
    """
    Run Kernel PCA experiment.

    Parameters
    ----------
    X_train : numpy.ndarray, 2d
        Training data
    X_test : numpy.ndarray, 2d
        Validation data set, with the same number of data dimensions
        as the training set
    kernel : str
        kernel function name
    d : int
        Maximum number of components
    sigma : float
        Kernel bandwidth parameter

    Returns
    -------
    fractions : numpy.ndarray, 1d
        The fraction of explained variance for each dimension

    """
    kernel_pca = KernelPCA(kernel=kernel, sigma=sigma, n_components=d)

    t0 = time.process_time()
    kernel_pca.fit_transform(X_train)
    t_kpca = time.process_time() - t0

    variances = kernel_pca.transform(X_test).var(0)

    K = kernel_pca.kernel.matrix(X_test, demean=True)

    total_variance = np.trace(K) / X_test.shape[0]

    fractions = np.cumsum(variances) / total_variance

    return fractions, t_kpca


def run_linear_pca(X_train: np.ndarray,
                   X_test:  np.ndarray,
                   d:       int       ) -> np.ndarray:
    """
    Run Linear PCA experiment.

    Parameters
    ----------
    X_train : numpy.ndarray, 2d
        Training data
    X_test : numpy.ndarray, 2d
        Validation data set, with the same number of data dimensions
        as the training set
    d : int
        Maximum number of components

    Returns
    -------
    fractions : numpy.ndarray, 1d
        The fraction of explained variance for each dimension

    """
    variances = np.zeros(d)

    d = min(d, X_train.shape[1])

    linear_pca = PCA(n_components=d)

    linear_pca.fit_transform(X_train)

    variances[:d] = linear_pca.transform(X_test).var(0)

    total_variance = np.sum(X_test.var(0)) # Sum of variances across dimensions

    fractions = np.cumsum(variances) / total_variance

    return fractions


def run_sparse_pca(X_train: np.ndarray,
                   X_test:  np.ndarray,
                   d:       int       ) -> np.ndarray:
    """
    Run Sparse PCA experiment.

    Sparse PCA is NP-hard in the number of components so we return
    NaNs if this is larger than 25.

    Parameters
    ----------
    X_train : numpy.ndarray, 2d
        Training data
    X_test : numpy.ndarray, 2d
        Validation data set, with the same number of data dimensions
        as the training set
    d : int
        Maximum number of components

    Returns
    -------
    fractions : numpy.ndarray, 1d
        The fraction of explained variance for each dimension

    """
    variances = np.zeros(d)
    variances.fill(np.nan)

    if X_train.shape[1] > 25:
        return variances

    d1 = min(d, X_train.shape[1])

    for i in range(d1):
        sparse_pca = SparsePCA(n_components=i+1)
        sparse_pca.fit_transform(X_train)
        variances[i] = sparse_pca.transform(X_test).var(0).sum()

    total_variance = np.sum(X_test.var(0))

    fractions = variances / total_variance

    fractions[np.isnan(fractions)] = max(fractions)

    return fractions


def run_lle(X_train: np.ndarray,
            X_test:  np.ndarray,
            d:       int       ) -> np.ndarray:
    """
    Run Locally-Linear Embedding experiment.

    The maximum possible number of components is the smallest of the
    number of data dimensions and one minus the number of data points.

    Parameters
    ----------
    X_train : numpy.ndarray, 2d
        Training data
    X_test : numpy.ndarray, 2d
        Validation data set, with the same number of data dimensions
        as the training set
    d : int
        Maximum number of components

    Returns
    -------
    fractions : numpy.ndarray, 1d
        The fraction of explained variance for each dimension

    """
    variances = np.zeros(d)
    variances.fill(np.nan)

    d = min(d, X_train.shape[1])

    for i in range(d):
        lle = LocallyLinearEmbedding(n_components=i+1)
        lle.fit(X_train)
        variances[i] = lle.transform(X_test).var(0).sum()

    # Find total variance by calculating the variance captured
    # by the maximum number of components
    max_dim = min(X_train.shape[0]-1, X_train.shape[1])
    lle = LocallyLinearEmbedding(n_components=max_dim)
    lle.fit(X_train)
    total_variance = lle.transform(X_test).var(0).sum()

    variances /= total_variance

    return variances


def run_ica(X_train: np.ndarray,
            X_test:  np.ndarray,
            d:       int       ) -> np.ndarray:
    """
    Run Indendent Components Analysis (ICA).

    For a too large number of components the algorithm will not converge,
    in which case we decrease the number of dimensions.

    Parameters
    ----------
    X_train : numpy.ndarray, 2d
        Training data
    X_test : numpy.ndarray, 2d
        Validation data set, with the same number of data dimensions
        as the training set
    d : int
        Maximum number of components

    Returns
    -------
    fractions : numpy.ndarray, 1d
        The fraction of explained variance for each dimension

    """
    variances = np.zeros(d)

    # ICA can not find more components than the number of data
    # dimensions or data points minus one
    max_dim = min(X_train.shape[0] - 1, X_train.shape[1])
    max_d   = min(max_dim, d)
    for i in range(max_d):
        ica = FastICA(n_components=i+1, tol=10e-3, max_iter=int(5000))
        try:
            ica.fit(X_train)
            variances[i] = ica.transform(X_test).var(0).sum()
        except: # pylint: disable=bare-except
            variances[i] = variances[i-1]

    # Find the total variance by calculating the variance captured with the
    # max number of components. It's hard to know what is the largest number
    # for the algorithm to converge, so we try different ones
    while 1:
        ica = FastICA(n_components=max_dim, tol=10e-3, max_iter=int(1000))
        try:
            ica.fit(X_train)
        except: # pylint: disable=bare-except
            max_dim -= 1
        else:
            break

    total_variance = ica.transform(X_test).var(0).sum()

    variances /= total_variance

    return variances


description = "Comparison of unsupervised methods."
parser = argparse.ArgumentParser(description=description,
                                 parents=[base_parser])

parser.add_argument('--ica', action='store_true',
                    help="include ICA, which can be time-consuming")

parser.add_argument('--insample', action='store_true',
                    help="train and test methods on the same data")


if __name__ == '__main__':

    args = parser.parse_args()

    sys.exit(main(**vars(args)))

