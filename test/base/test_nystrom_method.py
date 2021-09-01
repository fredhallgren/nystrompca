
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
from numpy.testing import assert_, assert_array_equal

from nystrompca.base import NystromMethod


def test_nystrom_method_init():

    nystrom = NystromMethod(m_subset=33)

    assert_(nystrom.m == 33)
    assert_(nystrom.subset is None)


def test_create_subset1():

    nystrom = NystromMethod(m_subset=10)

    nystrom.create_subset(10)

    assert_array_equal(nystrom.subset, np.arange(10))


def test_create_subset2():

    nystrom = NystromMethod(m_subset=5)

    nystrom.create_subset(10)

    assert_(len(set(np.arange(10)) - set(nystrom.subset)) == 5)

