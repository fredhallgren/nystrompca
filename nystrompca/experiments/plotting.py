
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
Plotting functions for the output of the experiments.

All text is in the *Times New Roman* font

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_context("paper")
sns.set(style="darkgrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})



def plot_results(results: pd.DataFrame) -> None:
    """
    Plot the results of the PCA experiments.

    Creates four line plots with the values for the bound,
    the difference between the Nyström and standard reconstruction
    error, and the difference between the subset PCA reconstruction
    error and the true one.

    Parameters
    ----------
    results : pandas.DataFrame
        Experimental results. The data is divided into different
        plots by the **dataset** column

    """
    if 'NYSTROMPCA_NOPLOT' in os.environ.keys():
        return

    plot_vars = ['Nyström diff.','Subset diff.','Conf. bound']

    results = pd.melt(results[['d','dataset'] + plot_vars], ['d','dataset'])

    grid = sns.FacetGrid(results, col='dataset', col_wrap=2, hue='variable',
                         hue_kws={'color':  [(0.208, 0.388, 0.517),
                                             (0.213, 0.481, 0.673),
                                             (0.498, 0.498, 0.498)],
                                  'marker': ['o','s','.'],
                                  'ls':     ['-', '--', ':']})

    grid.map(plt.plot, 'd', 'value', alpha=0.8)

    grid.add_legend(title="\n", loc='upper right')
    grid.set_axis_labels("PCA dimension", "Errors")
    plt.show()


def plot_R_squared(R_squared: pd.DataFrame,
                   title:     str       = "",
                   ylabel:    str       = "") -> None:
    """
    Plot the R-squared from the regression experiments.

    Parameters
    ----------
    R_squared : pandas.DataFrame
        R-squared from multiple regressions
    title : str
        Plot title
    ylabel : str
        Label for y-axis

    """
    if 'NYSTROMPCA_NOPLOT' in os.environ.keys():
        return

    sns.set(font_scale=1.5)
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.2)

    sns.heatmap(R_squared, cmap="YlGnBu")
    plt.title(title)
    plt.xlabel("Subset size")
    plt.ylabel(ylabel)
    plt.show()


def plot_targets(y:           np.ndarray,
                 predictions: np.ndarray,
                 title:       str       = "",
                 legend:      bool      = True) -> None:
    """
    Plot the predictions versus the actual targets for one regression
    in a heat map.

    Parameters
    ----------
    y : numpy.ndarray, 1d
        Actual targets
    predictions : numpy.ndarray, 1d
        Predicted targets
    title : str
        Plot title
    legend : bool, default=True
        Whether to draw a legend

    """
    if 'NYSTROMPCA_NOPLOT' in os.environ.keys():
        return

    y = y.squeeze()
    predictions = predictions.squeeze()

    sns.set(font_scale=1.5)
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    # Same axis limits every time we create the plot.
    plt.ylim(102.5, 142.5)
    plt.xlim(102.5, 142.5)

    xy = np.array([y.min(), y.max()])
    plt.plot(xy, xy, color='gray', alpha=1, linewidth=3, linestyle='--')
    sns.scatterplot(x=y, y=predictions, s=100)
    plt.xlabel("Actual target")
    plt.ylabel("Predicted target")
    plt.title(title)

    if legend:
        plt.legend(['Oracle'])

    plt.show()

