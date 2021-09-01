
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
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)

from nystrompca import NystromKPCA
from nystrompca.base import Kernel


def test_init():
    """
    Check that all expected variables are set or created in the constructor

    """
    nkpca = NystromKPCA(n_components = 10,
                        m_subset     = 20,
                        kernel       = 'laplace',
                        sigma        = 3,
                        degree       = 4,
                        coef0        = 2)

    assert nkpca.n_components == 10
    assert nkpca.m            == 20
    assert nkpca.kernel.name  == 'laplace'
    assert nkpca.demean

    assert nkpca.explained_variance_ is None
    assert nkpca.all_variances       is None
    assert nkpca.components_         is None
    assert nkpca.scores_             is None
    assert nkpca.errors_             is None
    assert nkpca.subset              is None
    assert nkpca.X                   is None
    assert nkpca.K_mm_p              is None
    assert nkpca.K_nm_p              is None
    assert nkpca.K_nm                is None


def test_fit_transform():
    """
    test the "fit_transform" method with the data matrix as
    the identity and a linear kernel

    """
    X = np.eye(3)

    m = 2

    nkpca = NystromKPCA(n_components=1, m_subset=m,
                        kernel='linear', scale=False)

    nkpca.subset = np.array([0, 1])

    nkpca.fit_transform(X)

    assert_array_almost_equal(nkpca.explained_variance_, np.array([1/3]))

    assert_array_almost_equal(np.abs(nkpca.components_),
                              np.array([[1 / np.sqrt(2)],
                                        [1 / np.sqrt(2)]]))

