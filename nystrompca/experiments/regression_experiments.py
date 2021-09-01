
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
Experiments to illustrate Nyström kernel PCR and comparing it to
Nyström kernel ridge regression.

"""

# Built-in modules
import os
import sys
import argparse
from typing import Union, Tuple
from itertools import product

# Third-party packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# This package
from nystrompca import NystromKPCR, NystromKRR
from nystrompca.experiments import (base_parser, get_airfoil_data,
                                    plot_R_squared, plot_targets)
from nystrompca.utils import logger


def main(kernel: str   = 'rbf',
         n:      int   = 1000,
         m:      int   = 100,
         d:      int   = 10,
         gamma:  float = 1,
         seed:   int   = None,
         noplot: bool  = False) -> int:
    """
    Run experiments for Nyström kernel PCR.

    Parameters
    ----------
    kernel : str, default 'rbf'
        Kernel function to use
    n : int, default 1000
        Total size of dataset
    m : int, default 100
        Size of subsampled dataset
    d : int, default 10
        Maximum subspace dimension
    gamma : float, default 1
        Ridge regularization parameter
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

    logger.header("Regression experiments")

    logger.info(f"n: {n}")

    X, y = get_airfoil_data(n)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=seed)

    run_multiple_regression_experiment(X_train, X_test, y_train, y_test,
                                       kernel, seed)

    run_single_regression_experiment(X_train, X_test, y_train, y_test,
                                     kernel, m, d, gamma, seed)

    return 0


def run_multiple_regression_experiment(X_train: np.ndarray,
                                       X_test:  np.ndarray,
                                       y_train: np.ndarray,
                                       y_test:  np.ndarray,
                                       kernel:  str       ,
                                       seed:    int       ) -> None:
    """
    Calculate and plot the R-squared across different parameters values.

    Calculate the regression for different subset sizes m and
    PCA dimensions d and plot the R-squared in a heat map.

    """
    logger.subheader("Experiment I: Multiple regressions R-squared")

    m_values = np.arange(10, 20)
    d_values = np.arange(11, 20)

    R_squared = pd.DataFrame(np.zeros((len(d_values), len(m_values))),
                             index=d_values, columns=m_values)

    for (m, d) in product(m_values, d_values):

        nystrom_kpcr = NystromKPCR(kernel=kernel, n_components=d,
                                   m_subset=m, sigma=1, seed=seed)

        R_squared.loc[d, m], _ = run_regression(nystrom_kpcr,
                                                X_train, X_test,
                                                y_train, y_test)

    R_squared.sort_index(ascending=False, inplace=True)

    plot_R_squared(R_squared, "Nyström KPCR", "PCA dimension")

    g_values = 1 / np.power(10, np.arange(5, 15))
    R_squared = pd.DataFrame(np.zeros((len(g_values), len(m_values))),
                             index=np.log10(g_values), columns=m_values)

    for (m, g) in product(m_values, g_values):

        nystrom_krr = NystromKRR(kernel=kernel, m_subset=m,
                                 gamma=g, sigma=1, seed=seed)

        R_squared.loc[np.log10(g), m] , _ = run_regression(nystrom_krr,
                                                           X_train, X_test,
                                                           y_train, y_test)

    plot_R_squared(R_squared, "Nyström KRR", "log of ridge parameter")


def run_single_regression_experiment(X_train: np.ndarray,
                                     X_test:  np.ndarray,
                                     y_train: np.ndarray,
                                     y_test:  np.ndarray,
                                     kernel:  str       ,
                                     m:       int       ,
                                     d:       int       ,
                                     gamma:   float     ,
                                     seed:    int       ) -> None:
    """
    Plot the predictions for one regression.

    Run one regression and plot the predicted target values versus
    the actual ones.

    """
    logger.subheader("Experiment II: Single regression predictions")

    kpcr = NystromKPCR(kernel=kernel, n_components=d, m_subset=m,
                       sigma=1, seed=seed)

    krr = NystromKRR(kernel=kernel, m_subset=m, gamma=gamma,
                     sigma=1, seed=seed)

    R_squared, predictions = run_regression(kpcr, X_train, X_test,
                                                  y_train, y_test)

    logger.info(f"R squared KPCR: {R_squared:.3f}")

    plot_targets(y_test, predictions, "Nyström KPCR", True)

    R_squared, predictions = run_regression(krr, X_train, X_test,
                                                 y_train, y_test)


    logger.info(f"R squared KRR: {R_squared:.3f}")

    plot_targets(y_test, predictions, "Nyström KRR", False)


def run_regression(model:   Union[NystromKPCR, NystromKRR],
                   X_train: np.ndarray,
                   X_test:  np.ndarray,
                   y_train: np.ndarray,
                   y_test:  np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Run one regression and calculate the R squared and predictions

    """
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    R_squared = model.score(X_test, y_test)

    return R_squared, predictions


description = "Run experiments for Nyström kernel PCR."
parser = argparse.ArgumentParser(description=description,
                                 parents=[base_parser])

parser.add_argument('-g', '--gamma', default=1, type=float,
                    help="ridge parameter")


if __name__ == '__main__':

    args = parser.parse_args()

    sys.exit(main(**vars(args)))

