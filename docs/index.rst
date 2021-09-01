
Nyström kernel PCA
==================

Welcome to the code documentation for the Python package ``nystrompca``. The package implements an efficient non-linear PCA and PCR as described in the paper `Kernel PCA with the Nyström method <https://www.nystrompca.com/>`_ including a confidence bound on its accuracy. It also contains experiments with real-world data to illustrate the methods and bound. The package also implements four related methods:

* Kernel PCA
* Kernel PCR
* Kernel Ridge Regression
* Nyström Kernel Ridge Regression.


Install
-------

Install the package through::

    $ pip install nystrompca


Source
------

The source code is available at `<https://github.com/cfjhallgren/nystrompca>`_.


License
-------

The software is released under Apache 2.0.


Usage
-----

Nyström kernel PCA
~~~~~~~~~~~~~~~~~~

::

    import numpy as np

    from nystrompca import NystromKPCA

    X = np.random.randn(1000, 100)

    pca = NystromKPCA(n_components=5, m_subset=10, kernel='rbf')
    pca.fit_transform(X)

    X_new = np.random.randn(100, 100)
    X_transformed = pca.transform(X_new)


Nyström kernel PCR
~~~~~~~~~~~~~~~~~~

::

    import numpy as np

    from nystrompca import NystromKPCR

    X = np.random.randn(1000, 100)
    y = np.random.randn(1000, 1)
    x_test = np.random.randn(1, 100)

    pcr = NystromKPCR(n_components=5, m_subset=10, kernel='rbf')
    pcr.fit(X, y)
    prediction = pcr.predict(x_test)


Confidence bound
~~~~~~~~~~~~~~~~

::

    import numpy as np

    from nystrompca import Kernel, calculate_bound

    X = np.random.randn(100, 10)

    kernel = Kernel('rbf')
    K_mm = kernel.matrix(X)

    bounds = calculate_bound(K_mm, n=1000, B=kernel.get_bound())


Experiments
~~~~~~~~~~~

Run the different experiments with the ``nystrompca`` command-line tool::

    $ nystrompca -h

    usage: nystrompca [-h] {methods,bound,regression} ...

    Run the different Nyström kernel PCA experiments.

    optional arguments:
      -h, --help            show this help message and exit

    available subcommands:
      {methods,bound,regression}
                            which experiment to run

    Display subcommand options with nystrompca <subcommand> -h


.. toctree::
   :maxdepth: 2
   :caption: Contents

   experiments/experiments
   algorithms/algorithms
   base/base
   utils/utils


