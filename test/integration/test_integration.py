
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


import numpy as np
from numpy.testing._private.utils import assert_array_less
from sklearn.preprocessing import scale
from numpy.testing import (assert_, assert_almost_equal,
                           assert_array_almost_equal, assert_equal)

from nystrompca import NystromKPCA, NystromKPCR, KernelPCA, KernelPCR
from nystrompca.base import Kernel
from nystrompca.experiments.data import *
from nystrompca.experiments.methods_experiments import (run_nystrom_kpca,
                                                        run_kernel_pca)
from nystrompca.experiments.bound_experiments import run_one_experiment
from nystrompca.utils import (get_inverse, get_eigendecomposition,
                              demean_matrix, get_kappa, flip_dimensions)


np.set_printoptions(precision=3, suppress=True)


def test_same_variance_nystrom_subset():
    """
    Check that Nyström PCA and Subset PCA capture the same amout of
    variance when m == d, as known from Theorem 2.

    """
    n = 100
    m = 10
    d = 10

    def run_test(X):

        nkpca = NystromKPCA(m_subset=m, n_components=d)
        nkpca.fit_transform(X)

        nystrom_variance = np.sum(nkpca.explained_variance_)

        subset_errors   = nkpca.get_subset_errors()
        tot_variance    = np.trace(nkpca.K_p) / n
        subset_variance = tot_variance - subset_errors[-1]

        assert_almost_equal(nystrom_variance, subset_variance)

    run_test(get_magic_data(n))
    run_test(get_segmentation_data(n))
    run_test(get_digits_data(n))


def test_nystrom_accuracy():
    """
    Check that Nystrom kernel PCA is always in between subset PCA and standard
    kernel PCA when using the same dataset for training and testing

    """
    n = 100
    m = 20
    d = 10

    def run_test(X):

        X = scale(X)

        subset, nystrom, _, sigma = run_nystrom_kpca(X, X, 'rbf', m, d, None)

        kpca, _ = run_kernel_pca(X, X, 'rbf', d, sigma)

        assert_array_less(subset, nystrom)
        assert_array_less(nystrom, kpca)

    run_test(get_yeast_data(n))
    run_test(get_digits_data(n))
    run_test(get_nips_data(n))



def test_same_errors_full_subset():
    """
    Check that if m = n then all errors are the same - the full dataset errors,
    the Nyström errors, and the subset errors; and that the Nyström errors are
    the same as the Nyström variances.

    """
    n = 100
    m = 100
    d = 20

    def run_test(X):
        results = run_one_experiment('rbf', X, m, d, alpha=0.9,
                                     samples=1, seed=None)
        assert_array_almost_equal(results["True errors"],
                                  results["Nyström errors"], 4)
        assert_array_almost_equal(results["True errors"],
                                  results["Subset errors"],  4)
        assert_array_almost_equal(results["Nyström PCA"],
                                  results["Nyström errors"], 4)

    # Random data
    X = np.random.randn(100,10)
    run_test(X)

    # Actual dataset
    X = get_magic_data(n)
    run_test(X)


def test_same_errors_full_pca():
    """
    Check that the Nyström and subset reconstruction errors are the same
    when d=m and when we don't centre the data.

    """
    n = 100
    m = 20
    d = 20

    def run_test(X):
        results = run_one_experiment('rbf', X, m, d, alpha=0.9,
                                     samples=1, seed=None)
        assert_almost_equal(results["Nyström errors"].iloc[-1],
                            results["Subset errors"].iloc[-1])

    # Random data
    X = np.random.randn(n,10)
    run_test(X)

    # Actual dataset
    X = get_magic_data(n)
    run_test(X)


def test_nystrom_pca_total_error():
    """
    Check that the sum of all Nyström eigenvalues equal the total variance
    within the subspace from the subset of data points.

    """
    n = 100
    m = 10

    nkpca = NystromKPCA(m_subset=m)

    def run_test(X):
        nkpca.fit_transform(X)

        nystrom_variance = np.sum(nkpca.explained_variance_)

        K_tilde          = nkpca.K_nm @ get_inverse(nkpca.K_mm) @ nkpca.K_nm.T
        K_tilde_p        = demean_matrix(K_tilde)
        subset_variance  = np.trace(K_tilde_p) / n

        assert_almost_equal(nystrom_variance, subset_variance)

    run_test(get_magic_data(n))
    run_test(get_drug_data(n))
    run_test(get_digits_data(n))


def test_subset_errors():
    """
    Test that the explicitly calculated subset error is the
    same as the last subset PCA error for d = m.

    """
    n = 100
    m = 20

    X = get_yeast_data(n)

    nkpca = NystromKPCA(m_subset=m)
    nkpca.fit_transform(X)

    X = scale(X)

    K = nkpca.kernel.matrix(X, demean=True)

    K_tilde = nkpca.K_nm @ get_inverse(nkpca.K_mm) @ nkpca.K_nm.T
    subset_var = np.trace(demean_matrix(K_tilde)) / n

    subset_error = np.trace(K) / n - subset_var

    subset_pca_errors = nkpca.get_subset_errors()

    assert_almost_equal(subset_pca_errors[-1], subset_error)


def test_subset_pca():
    """
    Check that the subset PCA reconstruction error from the dataset used to fit 
    the method is the same as the total variance minus the sum of the variances
    captured when we use the same dataset for these (this corresponds to checking
    that the two quantities in Theorem 3 in the paper calculate the same thing)

    """
    n = 100
    m = 20

    X = get_yeast_data(n)

    nkpca = NystromKPCA(m_subset=m)
    nkpca.fit_transform(X)

    X = scale(X)

    subset_pca_errors = nkpca.get_subset_errors()[:m]

    subset_pca_variances = np.zeros(m)
    tot_var = np.trace(nkpca.kernel.matrix(X, demean=True)) / n
    for i in range(m):
        subset_pca_variances[i] = nkpca.get_subset_variance(X, i)
    subset_pca_variances = tot_var - np.cumsum(subset_pca_variances)

    assert_almost_equal(subset_pca_errors, subset_pca_variances)


def test_standard_kpca():
    """
    If n = m then the principal scores and values of standard and
    Nyström kernel PCA should be the same.

    """
    n = 8
    m = 8

    X = get_yeast_data(n)

    nkpca = NystromKPCA(m_subset=m)
    nkpca.fit_transform(X)

    X = scale(X)

    kernel = Kernel()

    K = kernel.matrix(X, demean=True)

    L, U = get_eigendecomposition(K / n)

    scores = U @ np.diag(np.sqrt(L) * np.sqrt(n))

    scores, _ = flip_dimensions(scores, scores)

    assert_array_almost_equal(L, nkpca.explained_variance_)

    assert_array_almost_equal(nkpca.scores_, scores)


def test_linear_kernel():
    """
    With a the linear kernel for zero-mean data and m = n, the sum of
    eigenvalues should be same as the covariance matrix and the
    transformed scores should have the same norm as the original data.

    """
    n = 100
    m = 100

    nkpca = NystromKPCA(m_subset=m, kernel='linear')

    def run_test(X):

        X_transformed = nkpca.fit_transform(X)

        X = scale(X)

        # Equal eigenvalues
        C_n = X.T @ X / n
        L_C, _ = np.linalg.eigh(C_n)
        assert_almost_equal(L_C.sum(), nkpca.explained_variance_.sum(), 9)

        # Same norms
        norms0 = (X**2).sum(1)
        norms1 = (X_transformed**2).sum(1)

        assert_array_almost_equal(norms0, norms1)

    X = get_yeast_data(n)
    run_test(X)

    X = get_segmentation_data(n)
    run_test(X)


def test_scores_transform():
    """
    Check that the score calculated in the 'fit_transform' method are
    the same as calling the 'transform' method on the original data.

    """
    n = 100
    m = 20

    nkpca = NystromKPCA(m_subset=m)

    def run_test(X):

        scores_1 = nkpca.fit_transform(X)

        scores_2 = nkpca.transform(X)

        assert_array_almost_equal(scores_1, scores_2)

    X = get_yeast_data(n)
    run_test(X)

    X = get_cardiotocography_data(n)
    run_test(X)


def test_scores_transform_noncentred():
    """
    Check that the score calculated in the 'fit_transform' method are
    the same as calling the 'transform' method on the original data.

    """
    n = 100
    m = 20

    nkpca = NystromKPCA(m_subset=m, demean=False)

    def run_test(X):

        scores_1 = nkpca.fit_transform(X)

        scores_2 = nkpca.transform(X)

        assert_array_almost_equal(scores_1, scores_2)

    X = get_yeast_data(n)
    run_test(X)

    X = get_cardiotocography_data(n)
    run_test(X)


def test_scores_orthogonal():
    """
    Check that the scores are orthogonal (uncorrelated)

    """
    n = 100
    m = 20

    nkpca = NystromKPCA(m_subset=m)

    def run_test(X):

        scores = nkpca.fit_transform(X)

        d = scores.shape[1]
        cov = scores.T @ scores
        assert_array_almost_equal(cov - np.diag(np.diag(cov)), np.zeros((d,d)))

    X = get_magic_data(n)
    run_test(X)

    X = get_cardiotocography_data(n)
    run_test(X)


def test_kernel_pca_comparison():
    """
    Check that Nyström kernel PCA gives the same results
    as standard kernel PCA when n = m.

    """
    n = 100
    m = 100
    d = 10

    nkpca = NystromKPCA(m_subset=m, n_components=d)

    kpca = KernelPCA(n_components=d)

    def run_test(X):

        nkpca.fit_transform(X)
        kpca.fit_transform(X)

        assert_array_almost_equal(nkpca.scores_, kpca.scores_)
        assert_array_almost_equal(nkpca.explained_variance_,
                                  kpca.explained_variance_)


    X = get_magic_data(n)
    run_test(X)

    X = get_yeast_data(n)
    run_test(X)


def test_kernel_pca_scores_transform():
    """
    Check that the score calculated in the 'fit_transform' method are
    the same as calling the 'transform' method on the original data.

    """
    n = 100

    kpca = KernelPCA()

    def run_test(X):

        scores_1 = kpca.fit_transform(X)

        scores_2 = kpca.transform(X)

        assert_array_almost_equal(scores_1, scores_2)

    X = get_yeast_data(n)
    run_test(X)

    X = get_cardiotocography_data(n)
    run_test(X)


def test_kernel_pca_scores_transform_noncentred():
    """
    Check that the score calculated in the 'fit_transform' method are
    the same as calling the 'transform' method on the original data.

    """
    n = 100

    kpca = KernelPCA(demean=False)

    def run_test(X):

        scores_1 = kpca.fit_transform(X)

        scores_2 = kpca.transform(X)

        assert_array_almost_equal(scores_1, scores_2)

    X = get_yeast_data(n)
    run_test(X)

    X = get_cardiotocography_data(n)
    run_test(X)


def test_get_kappa_tilde():
    """
    Check that the calculation of kappa tilde using the original kernel
    matrix just gives the centred one
    """
    n = 100
    m = 100

    X = get_segmentation_data(n)
    kernel = Kernel('cauchy', sigma=100)
    K = kernel.matrix(X, demean=False)

    K1 = demean_matrix(K.copy())

    nkpca = NystromKPCA(m_subset=m, scale=False, kernel='cauchy', sigma=100)
    nkpca.fit_transform(X)

    K2 = nkpca.get_kappa_p(K)

    assert_array_almost_equal(K1, K2)


def test_get_kappa():
    """
    Check that calculation of kappa prime using the original kernel
    matrix everywhere gives just the demeaned kernel matrix.

    """
    n = 100
    X = get_segmentation_data(n)
    kernel = Kernel('cauchy')
    K = kernel.matrix(X)

    K1 = demean_matrix(K)

    K2 = get_kappa(K, K)

    assert_array_almost_equal(K1, K2)


def test_kernel_pcr_comparison():
    """
    Check that Nyström kernel PCR gives the same results as standard
    kernel PCR when n = m.

    """
    n = 100
    m = 100
    d = 10

    nkpcr = NystromKPCR(m_subset=m, n_components=d)

    kpcr = KernelPCR(n_components=d)

    X, y = get_airfoil_data(n)

    nkpcr.fit(X, y)
    kpcr.fit(X, y)

    assert_almost_equal(nkpcr.alpha, kpcr.alpha)
    assert_array_almost_equal(nkpcr.beta, kpcr.beta)
    assert_almost_equal(nkpcr.score(X, y), kpcr.score(X, y))


def test_predictions():
    """
    Check that the predictions on the same dataset used for fitting
    have R squared between 0 and 1, as well as the same mean predictions.

    """
    n = 100
    m = 20
    d = 10

    X, y = get_airfoil_data(n)

    nystrom_kpcr = NystromKPCR(n_components=d, m_subset=m)

    nystrom_kpcr.fit(X, y)

    predictions = nystrom_kpcr.predict(X)

    R2 = nystrom_kpcr.score(X, y)

    assert_almost_equal(predictions.mean(), y.mean(), 6)

    assert R2 >= 0 and R2 <= 1
