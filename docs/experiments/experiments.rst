experiments
===========

This module contains experiments to illustrate the implemented algorithms.

There are experiments for the accuracy of Nyström kernel PCA and the confidence bound (`bound_experiments.py`), for Nyström kernel PCR compared to kernel ridge regression with the Nyström method (`regression_experiments.py`), and a comparison of Nyström kernel PCA versus a range of unsupervised learning techniques (`methods_experiments.py`).

The plotting functions and data access functions are in separate files.

This module also contains the entry-point for the command-line utility to run the experiments in `console_script.py` as well as a  Here the command-line utility is also


.. toctree::
   :maxdepth: 1
   :caption: Contents

   methods_experiments
   bound_experiments
   regression_experiments
   plotting
   data
   console_script
   base_parser
