
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
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_equal)

from nystrompca.base import Kernel


def test_init():

    k = Kernel(kernel='poly', sigma=1, degree=3, coef0=5, normalize=False)

    assert k.sigma     == 1
    assert k.degree    == 3
    assert k.coef0     == 5
    assert k.normalize == False


def test_rbf():

    k = Kernel('rbf', sigma=2)

    assert_equal(k.get_name(), 'rbf')

    assert_equal(k(np.array([3]), np.array([3])), 1)
    assert_almost_equal(k(np.array([5, 3]), np.array([4, 1])), 0.287, 3)

    assert_equal(k.get_bound(), 1)


def test_poly():

    k1 = Kernel('poly', degree=1, normalize=True, coef0=0)

    assert_equal(k1.get_name(), 'poly')
    assert_equal(k1(np.array([3]), np.array([3])), 1)
    assert_equal(k1.get_bound(), 1)


    k2 = Kernel('poly', normalize=False)
    assert_equal(k2(np.array([5, 3]), np.array([4, 1])), 576)
    assert_equal(k2.get_bound(), np.inf)


def test_matrix():

    k = Kernel('linear')

    X1 = np.arange(1, 4)[:,np.newaxis]
    X2 = np.arange(1, 3)[:,np.newaxis]

    matrix = k.matrix(X1, X2, demean=False)

    expected = np.array([[1, 2],
                         [2, 4],
                         [3, 6]])

    assert_array_equal(matrix, expected)


def test_sigma():

    k = Kernel('rbf')

    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])

    k.calc_sigma(X)

    assert_almost_equal(k.sigma, np.sqrt(8))

