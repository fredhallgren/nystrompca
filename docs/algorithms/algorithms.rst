algorithms
==========

This module contains the proposed methods for Nyström kernel PCA and PCR as well as a number of other methods used for comparisons. It also includes calculation of the confidence bound.

Each algorithm is a class and has an API similar to `scikit-learn`. If it's a dimensionality reduction method it has the member functions `fit_transform` to fit the model using training data and `tranform` to transform new data according to the fitted model. If it's a supervised method it has member functions `fit` to fit the method, `predict` to create predictions for new data, and `score` to calculate an evaluation metric (R-squared)for given values of the predictor and target variables.

The algorithms inherit from the base classes in the module `base/`. A UML diagram for the inheritance relationships is as follows 

.. image:: ../../img/classes_nystrompca.png
  :alt: UML diagram

.. toctree::
   :maxdepth: 5
   :caption: Contents

   nystrom_KPCA
   nystrom_KPCR
   kernel_PCA
   kernel_PCR
   kernel_RR
   nystrom_KRR
   bound
