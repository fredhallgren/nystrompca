
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
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from nystrompca import KernelPCA


def test_init():

    kpca = KernelPCA(n_components = 10,
                     kernel       = 'poly',
                     sigma        = 3,
                     degree       = 4,
                     coef0        = 2,
                     demean       = False,
                     scale        = False)

    assert kpca.n_components == 10
    assert kpca.kernel.name  == 'poly'
    assert kpca.demean      == False

    assert kpca.explained_variance_ is None
    assert kpca.all_variances       is None
    assert kpca.components_         is None
    assert kpca.scores_             is None
    assert kpca.errors_             is None
    assert kpca.K_p                 is None
    assert kpca.Q                   is None
    assert kpca.X                   is None


def test_fit_transform():

    kpca = KernelPCA(n_components = 2,
                     kernel       = 'poly',
                     sigma        = 3,
                     degree       = 2,
                     coef0        = 0,
                     normalize    = False,
                     demean       = False,
                     scale        = False)

    X = np.array([[1,  2, 1],
                  [0, -1, 1],
                  [2, 1,  0]])

    actual_scores = kpca.fit_transform(X)

    assert_array_almost_equal(kpca.explained_variance_,
                              [15.821, 4.529], decimal=3)

    actual_scores = actual_scores.tolist()
    actual_scores.sort(key = lambda x: x[0])
    assert_array_almost_equal(actual_scores, [[0.221,  0.089],
                                              [4.001,  2.997],
                                              [5.604, -2.144]], decimal=3)


def test_transform():

    kpca = KernelPCA(n_components = 2,
                     kernel       = 'poly',
                     sigma        = 3,
                     degree       = 2,
                     coef0        = 0,
                     normalize    = False,
                     demean       = False,
                     scale        = False)

    X = np.array([[1,  2, 1],
                  [0, -1, 1],
                  [2, 1,  0]])

    kpca.fit_transform(X)

    X_new = np.array([[ 1,1,0],
                      [-1,0,1]])

    X_transformed = kpca.transform(X_new)

    assert_array_almost_equal(X_transformed, [[1.826, 0.572],
                                              [0.342, 0.889]], decimal=3)

