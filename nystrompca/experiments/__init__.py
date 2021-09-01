
"""
Experiments to illustrate Nyström kernel PCA, the confidence bound
and Nyström kernel PCR.

"""

from .plotting import plot_results, plot_R_squared, plot_targets
from .data import (get_magic_data, get_yeast_data, get_cardiotocography_data,
                   get_segmentation_data, get_airfoil_data)
from .base_parser import base_parser
from . import (methods_experiments, bound_experiments, regression_experiments,
               plotting, data)
