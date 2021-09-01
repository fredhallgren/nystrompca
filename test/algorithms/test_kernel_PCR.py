
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
from numpy.testing import assert_equal, assert_array_equal

from nystrompca import KernelPCR


def test_predict_score():

    kpcr = KernelPCR(kernel='linear', normalize=False,
                     scale=False, demean=False)

    kpcr.fit(np.eye(3), np.ones(3)) 

    kpcr.alpha = 3
    kpcr.beta  = np.array([[1,2,3]]).T
    X_new      = np.array([[1,1,1],
                           [3,2,1]])
    
    assert_array_equal(kpcr.predict(X_new), [[9],[17]])
    
    assert_equal(kpcr.score(X_new, np.array([10, 15])), 0.6)

