
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
Experiments to evaluate Nyström kernel PCA and the confidence bound,
comparing it to standard kernel PCA and kernel PCA using the subset
of data points directly.

"""

# Built-in modules
import os
import sys
import argparse
from typing import Tuple
from multiprocessing import Pool, cpu_count

# Third-party packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# This package
from nystrompca import NystromKPCA, calc_conf_bound
from nystrompca.base import Kernel
from nystrompca.utils import logger, get_eigendecomposition, get_tail_sums
from nystrompca.experiments import base_parser, data, plot_results


# Always display numbers in decimal format, using 4 decimals
pd.options.display.float_format = '{:,.4f}'.format


def main(kernel:  str   = 'rbf',
         n:       int   = 1000,
         m:       int   = 100,
         d:       int   = 10,
         alpha:   float = 0.9,
         samples: int   = 1,
         seed:    int   = None,
         noplot: bool   = False) -> int:
    """
    Run the experiments four different datasets.

    After each plot is shown the program halts, close the plot to continue.

    Parameters
    ----------
    kernel: str
        Kernel function to use
    n : int
        Total size of dataset
    m : int
        Size of subsampled dataset
    d : int
        Maximum PCA dimension
    alpha : float
        Confidence level
    samples : int, default 1
        Number of subset samples to average
    seed : int, default None
        Random seed for the Nyström subset
    noplot: bool, default False
        If True then don't display plots

    Returns
    -------
    int
        Return code, 0 if success

    """
    if noplot:
        os.environ['NYSTROMPCA_NOPLOT'] = "NOPLOT"

    logger.header("Confidence bound experiments")

    logger.info(f"Kernel: {kernel}")
    logger.info(f"n: {n}")
    logger.info(f"m: {m}")
    logger.info(f"samples: {samples}")
    logger.info(f"alpha: {alpha}")

    all_results = pd.DataFrame()

    for dataset in ('magic', 'yeast', 'cardiotocography', 'segmentation'):

        logger.subheader(f"Dataset: {dataset}")

        X = getattr(data, "get_" + dataset + "_data")(n)

        results = run_one_experiment(kernel, X, m, d, alpha, samples, seed)
        logger.info("Results:")
        logger.info(results[['d',
                             'Nyström diff.',
                             'Subset diff.',
                             'Conf. bound',
                             'Total variance']])

        results['dataset'] = dataset

        all_results = all_results.append(results)

    all_results.reset_index(drop=True, inplace=True)

    plot_results(all_results)

    return 0


def run_one_experiment(kernel:  str,
                       X:       np.ndarray,
                       m:       int,
                       d:       int,
                       alpha:   float,
                       samples: int,
                       seed:    int       ) -> pd.DataFrame:
    """
    Run an experiment for one kernel and dataset.

    Calculate the Nyström PCA bound, the actual difference between the
    Nyström and full reconstruction errors, and the difference between
    the full reconstruction error and the reconstruction error for the
    PCA subspace from the subset directly, for a number of different
    PCA subspace dimensions.

    The functions calculates the Nyström errors and bounds for multiple
    different subset samples and then takes the average. This is done
    in parallel using the maximum number of available CPU cores.

    Parameters
    ----------
    kernel : str
        Kernel function
    X : numpy.ndarray, 2d
        Data matrix, with observations in the rows
    m : int
        The size of the Nyström subset
    d : int
        Maximum PCA dimension
    alpha : float
        Confidence level
    samples : int
        Number of samples over which to calculate the Nyström quantities
    seed : int
        Random seed

    Returns
    -------
    results : pandas.DataFrame
        The results of one experiment

    """
    if samples == 1:
        results = get_nystrom_results(X, kernel, m, d, alpha, seed)

    else:
        with Pool(cpu_count()) as p:
            results = p.starmap(get_nystrom_results,
                                samples*[(X, kernel, m, d, alpha, seed)])

        results = pd.concat(results, axis=0)

        results = results.groupby('d', as_index=False).mean()

    true_errors, total_variance = get_true_errors(X, kernel, d)

    results['True errors']    = true_errors
    results['Nyström diff.']  = results['Nyström errors'] - true_errors
    results['Subset diff.']   = results['Subset errors']  - true_errors
    results['Total variance'] = total_variance

    # Some values may be zero within machine epsilon, set these to exactly zero
    results[np.abs(results) < 1e-14] = 0

    return results


def get_nystrom_results(X:      np.ndarray,
                        kernel: str,
                        m:      int,
                        d:      int,
                        alpha:  float,
                        seed:   int     ) -> pd.DataFrame:
    """
    Get experimental results for one Nyström PCA experiment, for one
    sampling of the subset.

    Calculates the reconstruction errors, explained variances,
    subset PCA reconstruction errors and the confidence bounds
    for multiple PCA dimensions.

    Parameters
    ----------
    X : numpy.ndarray, 2d
        Data matrix
    kernel : str
        Name of kernel function, supplied to the NystromKPCA class.
    m : int
        Subset size
    d : int
        Maximum PCA dimension
    alpha : float
        Confidence level
    seed : int
        Random seed

    Returns
    -------
    one_result : pandas.DataFrame
        Table with results, with columns 'd', 'Nyström errors', 'Nyström PCA',
        'Subset errors', 'Conf. bound', and with each row containing the
        results for one PCA dimension

    """
    n = X.shape[0]

    nystrom_kpca = NystromKPCA(kernel       = kernel,
                               n_components = d,
                               m_subset     = m,
                               demean       = False,
                               seed         = seed)

    nystrom_kpca.fit_transform(X)

    one_result = pd.DataFrame()

    one_result['d']              = np.arange(d) + 1
    one_result['Nyström errors'] = nystrom_kpca.get_reconstruction_errors()
    one_result['Nyström PCA']    = get_tail_sums(nystrom_kpca.all_variances, d)

    one_result['Subset errors']  = nystrom_kpca.get_subset_errors()

    one_result['Conf. bound']    = calc_conf_bound(nystrom_kpca.K_mm_p, n,
                                               nystrom_kpca.kernel.get_bound(),
                                               alpha)[:d]

    return one_result


def get_true_errors(X:          np.ndarray,
                    kernelname: str,
                    d:          int       ) -> Tuple[np.ndarray, float]:
    """
    Calculate the reconstruction error for standard kernel PCA

    Parameters
    ----------
    X : numpy.ndarray, 2d
        Data matrix
    kernel : str
        Name of kernel function, supplied to the NystromKPCA class.
    d : int
        Maximum PCA dimension

    Returns
    -------
    true_errors : numpy.ndarray
        Standard kernel PCA reconstruction errors for all PCA dimensions
    total_variance : float
        The total "variance" of the data (reconstruction error for
        an empty subspace)

    """
    n = X.shape[0]

    kernel = Kernel(kernelname)

    K = kernel.matrix(scale(X), demean=False)
    L, _ = get_eigendecomposition(K / X.shape[0])

    # Reconstruction errors for PCA on the entire dataset (for all d)
    true_errors = get_tail_sums(L, d)

    # Maximum reconstruction error
    total_variance = np.trace(K) / n

    return true_errors, total_variance # type: ignore[return-value]


description = "Calculate the confidence bound for different datasets."
parser = argparse.ArgumentParser(description=description,
                                 parents=[base_parser])

parser.add_argument('--alpha', default=0.9, type=float,
                    help="confidence level")

parser.add_argument('--samples', default=1, type=int,
                    help="number of subset samples to average")


if __name__ == '__main__':

    args = parser.parse_args()

    sys.exit(main(**vars(args)))

