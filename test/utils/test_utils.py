
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
from numpy.testing import (assert_, assert_array_equal,
                           assert_array_almost_equal)

from nystrompca.utils import *


def test_demean_matrix():

    M = np.array([[1, 2, 3],
                  [2, 4, 5],
                  [3, 5, 6]])

    expected = np.array([[ 4, -2, -2],
                         [-2,  1,  1],
                         [-2,  1,  1]]) / 9

    assert_array_equal(np.round(demean_matrix(M), 9),
                       np.round(expected, 9))


def test_get_inverse1():

    M = np.diag(np.arange(1,6))

    inv = get_inverse(M, np.sqrt)

    expected = np.array([1, 1/np.sqrt(2), 1/np.sqrt(3), 1/2, 1/np.sqrt(5)])

    assert_array_equal(inv, np.diag(expected))


def test_get_inverse2():

    M = np.array([[4, 2],
                  [2, 3]])

    expected = np.array([[ 3, -2],
                         [-2,  4]]) / 8

    assert_array_almost_equal(get_inverse(M), expected, 12)


def test_get_eigendecomposition():

    M = np.diag(np.arange(1,6))

    L, U = get_eigendecomposition(M)

    assert_array_equal(L, np.arange(5, 0, -1))

    assert_array_equal(U, np.eye(5)[::-1])


def test_get_tail_sums():

    L = np.array([5,4,3,2,1])

    tail_sums = get_tail_sums(L)

    assert_array_equal(tail_sums, np.array([10, 6, 3, 1, 0]))

    tail_sums = get_tail_sums(L, 3)

    assert_array_equal(tail_sums, np.array([10, 6, 3]))


def test_get_kappa_non_demean():

    kappa = np.random.randn(10,1)

    K = np.random.randn(10, 10)

    assert_array_equal(kappa, get_kappa(kappa, K, demean=False))


def test_get_kappa1():

    kappa = np.array([[1],
                      [2]])

    K_mn = np.array([[1, 2, 3],
                     [4, 5, 6]])

    actual = get_kappa(kappa, K_mn)

    expected = np.array([[ 1],
                         [-1]])

    assert_array_equal(actual, expected)


def test_get_kappa2():
    
    K = np.arange(6).reshape((2,3))

    kappa = np.arange(4).reshape((2,2))

    expected = np.array([[ 0.5,  0.5],
                         [-0.5, -0.5]])

    actual = get_kappa(kappa, K)

    assert_array_equal(actual, expected)


def test_to_column_vector():

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([[1, 2, 3]])

    expected = np.array([[1],
                         [2],
                         [3]])

    assert_array_equal(to_column_vector(arr1), expected)
    assert_array_equal(to_column_vector(arr2), expected)


def test_flip_dimensions():

    scores              = np.array([[-1, 1, -1],
                                    [-2, 2, -1],
                                    [-3, 3, -1]])

    components          = np.array([[-2, -5, 1],
                                    [-2,  2, 1],
                                    [-2, -1, 1]])

    expected_scores     = np.array([[1, 1, 1],
                                    [2, 2, 1],
                                    [3, 3, 1]])

    expected_components = np.array([[2, -5, -1],
                                    [2,  2, -1],
                                    [2, -1, -1]])

    actual_scores, actual_components = flip_dimensions(scores, components)

    assert_array_equal(actual_scores,     expected_scores)
    assert_array_equal(actual_components, expected_components)

