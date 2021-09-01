
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
End-to-end test that runs the bound experiments to ensure
successful completion

"""

from numpy.testing import assert_equal

from nystrompca.experiments.bound_experiments import main as bound_main
from nystrompca.experiments.methods_experiments import main as methods_main


def test_methods_experiments():

    return_code = methods_main(n=20, m=10, d=5, noplot=True)

    assert_equal(return_code, 0)


def test_bound_experiments():

    return_code = bound_main(n=100, m=10, d=5, noplot=True)

    assert_equal(return_code, 0)
