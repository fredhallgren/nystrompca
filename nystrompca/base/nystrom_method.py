
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
Base class for the Nyström method.

"""

import numpy as np


class NystromMethod:

    """
    Base class for algorithms using the Nyström method

    Parameters
    ----------
    m_subset : int, default 10
        Size of Nyström subset
    seed : int, default None
        Seed for the random number generator

    """
    def __init__(self, m_subset: int = 10,
                       seed:     int = None,
                       **kwargs            ):

        super().__init__(**kwargs) # type: ignore[call-arg]

        self.m = m_subset

        self.subset: np.ndarray = None

        self.seed = seed


    def create_subset(self, n: int) -> None:
        """
        Randomly select a subset of *m* integers without replacement from
        all integers up to *n*. Sorts the selected subset in increasing order.

        Parameters
        ----------
        n : int
            Total dataset size

        Raises
        ------
        ValueError
            If *m* > *n*

        """
        rng = np.random.default_rng(self.seed)

        subset = rng.choice(range(n), self.m, replace=False)

        self.subset = np.sort(subset) # Not necessary but easier to debug
