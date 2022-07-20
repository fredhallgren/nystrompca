
# Copyright 2022 Fredrik Hallgren
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


from nystrompca import calc_conf_bound, calculate_bound
from nystrompca.base import Kernel

import numpy as np
from numpy.testing import assert_array_almost_equal


def test_calc_conf_bound1():

    L     = np.arange(1,11)[::-1] * 10
    n     = 100
    alpha = 0.5
    B     = 1
    
    D = 2 * 0.9 * np.sqrt(2*np.log(4)) / np.sqrt(90)
    
    expected_bounds = np.cumsum(L[:-1]/10) * D**2 + D ** 3
    expected_bounds = np.r_[expected_bounds, np.nan]

    bounds = calc_conf_bound(np.diag(L), n, B, alpha)

    assert_array_almost_equal(bounds, expected_bounds)


def test_calc_conf_bound2():

    L     = np.exp(np.linspace(1,0,10))
    n     = 20
    alpha = 0.5
    B     = 10
    
    D = 10 * np.sqrt(2*np.log(4)) / np.sqrt(10)
    
    expected_bounds = np.cumsum(L[:-1]/10) + D
    expected_bounds = np.r_[expected_bounds, np.nan]

    bounds = calc_conf_bound(np.diag(L), n, B, alpha)

    assert_array_almost_equal(bounds, expected_bounds)


def test_calc_conf_bound3():

    L     = np.arange(1,11)[::-1] * 10
    n     = 1000
    alpha = 0.9
    B     = 0.5
    
    D = 0.99 * np.sqrt(2*np.log(20)) / np.sqrt(990)
    
    expected_bounds = np.cumsum(L[:-1]/10) * D**2 + D ** 3
    expected_bounds = np.r_[expected_bounds, np.nan]

    bounds = calc_conf_bound(np.diag(L), n, B, alpha)

    assert_array_almost_equal(bounds, expected_bounds)


def test_calculate_bound():

    X = np.random.randn(100,10)
    kernel = Kernel(kernel='cauchy', sigma=5)
    n = 2000
    alpha = 0.75
    K_mm = kernel.matrix(X, demean=False)
    bounds1 = calculate_bound(X, n, 'cauchy', alpha, sigma=5)
    bounds2 = calc_conf_bound(K_mm, n, 1, alpha)

    assert_array_almost_equal(bounds1, bounds2)

