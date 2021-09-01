
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
Abstract base class for dimensionality reduction methods with only
independent variables that calculates new data representations
in a different coordinate system.

"""

from abc import ABC, abstractmethod

import numpy as np


class Transformation(ABC):
    """
    Abstract base class for dimensionality reduction methods that
    transform the data.

    n_components : int, default None
        Number of latent dimensions to use, such as the number of
        principal components. Default is unspecified and depends on
        the particular method.

    Attributes
    ----------
    scores_ : numpy.ndarray, 2d
        Transformed data points, with each new data point in a row
        of the array. These are for example the principal scores.
        They should be created in the 'fit_transform' method.

    """
    def __init__(self, n_components: int = None, **kwargs):

        super().__init__(**kwargs) # type: ignore[call-arg]

        self.n_components:   int = n_components

        self.X:       np.ndarray = None

        self.n:              int = None

        self.scores_: np.ndarray = None


    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the data transformation

        Parameters
        ----------
        X: numpy.ndarray, 2d
            Input data matrix with each data point in a row

        Returns
        -------
        scores_ : numpy.ndarray
            Transformed data points with new data points along the rows,
            such as the principal scores.

        """
        ...


    @abstractmethod
    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted model

        Parameters
        ----------
        X_new : numpy.ndarray, 2d
            Data to transform, with one data point in each row.

        Returns
        --------
        X_transformed : numpy.ndarray
            Transformed data with same size as input data

        """
        ...


    def tot_variance(self, X: np.ndarray = None):
        """
        Calculate the sum of the variances across all dimensions
        in the data. Using n degrees of freedom.

        Parameters
        ----------
        X : numpy.ndarray, 2d
            Dataset

        Raises
        ------
        ValueError
            If no dataset exists

        """
        if X is None:
            X = self.X

        if X is None:
            raise ValueError("No dataset supplied")

        total_variance = np.sum(X.var(0))

        return total_variance

