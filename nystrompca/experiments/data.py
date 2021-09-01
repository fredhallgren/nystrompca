
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
This module contains function to read different datasets from the
UCI machine learning repository. The raw dataset are included in
the repository in the `data/` folder in the root directory, apart
from the `digits` dataset which can be downloaded with scikit-learn.

"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits


DATA_FOLDER = Path(__file__).parent.joinpath('../../data/')


def get_magic_data(n: int) -> np.ndarray:
    """
    Read the magic telescope data

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    data : numpy.ndarray
        Data matrix

    """
    data = []
    with open(DATA_FOLDER.joinpath("magic_gamma_telescope.dat")) as f:
        for line in f.readlines():
            split_line = line.split(',')[:-1]
            data.append(split_line)

    data_arr = np.asarray(data, dtype=np.float64)

    data_arr = data_arr[:n]

    return data_arr


def get_yeast_data(n: int) -> np.ndarray:
    """
    Read the yeast dataset with protein location sites for fungi

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    data : numpy.ndarray
        Data matrix

    """
    df = pd.read_csv(DATA_FOLDER.joinpath("yeast.dat"), header=None,
                     delimiter=r'\s+')

    df.drop(0, axis=1, inplace=True)

    df = pd.get_dummies(df)

    return df.values[:n]


def get_cardiotocography_data(n: int) -> np.ndarray:
    """
    Read the cardiotocography dataset with heart measurement data

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    data : numpy.ndarray
        Data matrix

    """
    data = []
    with open(DATA_FOLDER.joinpath("cardiotocography.dat")) as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            split_line = line.split('\t')[3:] # keep numeric data
            data.append(split_line)

    data_arr = np.asarray(data, dtype=np.float64)

    data_arr = data_arr[:n]

    return data_arr


def get_segmentation_data(n: int) -> np.ndarray:
    """
    Read the segmentation dataset with various data on images

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    data : numpy.ndarray
        Data matrix

    """
    data = []
    with open(DATA_FOLDER.joinpath("segmentation.dat")) as f:
        for line in f.readlines():
            split_line = line.split(' ')
            data.append(split_line)

    data_arr = np.asarray(data, dtype=np.float64)

    data_arr = data_arr[:n]

    return data_arr


def get_drug_data(n: int) -> np.ndarray:
    """
    Read the drug consumption dataset

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    data : numpy.ndarray
        Data matrix

    """
    data = pd.read_csv(DATA_FOLDER.joinpath("drug.dat"),
                       header=None, index_col=0)

    data = data.replace('CL0', 0)
    data = data.replace('CL1', 1)
    data = data.replace('CL2', 2)
    data = data.replace('CL3', 3)
    data = data.replace('CL4', 4)
    data = data.replace('CL5', 5)
    data = data.replace('CL6', 6).astype(float)

    return data.values[:n]


def get_digits_data(n: int) -> np.ndarray:
    """
    Read the digits dataset from UCI through the scikit-learn helper
    function. The dataset contains the flattened grayscale pixel values
    from 8x8 images of handwritten digits.

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    X : numpy.ndarray, 2d
        Independent variables

    """
    digits = load_digits()

    return digits['data'][:n]


def get_dailykos_data(n: int) -> np.ndarray:
    """
    Read the dailykos dataset with word frequencies of blog posts
    into bag-of-words vectors

    Parameters
    ----------
    n : int
        Maximum number of bag-of-words vectors

    Returns
    -------
    X : numpy.ndarray, 2d
        Independent variables

    """
    df = read_bag_of_words('dailykos.dat')

    X = df.values[:n]

    return X


def get_nips_data(n: int) -> np.ndarray:
    """
    Read the dailykos dataset with word frequencies in NIPS papers
    into bag-of-words vectors

    Parameters
    ----------
    n : int
        Maximum number of bag-of-words vectors

    Returns
    -------
    X : numpy.ndarray, 2d
        Independent variables

    """
    df = read_bag_of_words('nips.dat')

    X = df.values[:n]

    return X


def read_bag_of_words(dataset: str) -> pd.DataFrame:
    """
    Read word frequencies and convert them into bag-of-words vectors

    Parameters
    ----------
    dataset: str
        name of file to read from

    Returns
    -------
    df : pandas.DataFrame
        Bag-of-words vectors along the rows

    """

    df = pd.read_csv(DATA_FOLDER.joinpath(dataset),
                     names=['docID','wordID','count'], delimiter=' ')

    df = df.pivot(index='docID', columns='wordID', values='count')

    df.replace(to_replace=np.nan, value=0, inplace=True)

    return df


def get_airfoil_data(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the airfoil dataset with wind tunnel measurements from NASA.
    This dataset is used in the regression experiments

    Parameters
    ----------
    n : int
        Maximum number of data points

    Returns
    -------
    X : numpy.ndarray, 2d
        Independent variables
    y : numpy.ndarray, 1d
        Dependent variable

    """
    data = []
    with open(DATA_FOLDER.joinpath("airfoil.dat")) as f:
        for line in f.readlines():
            split_line = line.split('\t')
            data.append(split_line)

    data_arr: np.ndarray = np.asarray(data, dtype=np.float64)

    data_arr = data_arr[:n]

    X: np.ndarray = data_arr[:,:-1]
    y: np.ndarray = data_arr[:,-1]

    return X, y

