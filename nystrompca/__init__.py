
"""
Implementation of Nyström Kernel PCA and a confidence bound on its accuracy

Experiments to evaluate the methods and confidence bound

Also implements a few other methods used for comparison

"""

from .algorithms.kernel_RR      import KernelRR
from .algorithms.kernel_PCA     import KernelPCA
from .algorithms.nystrom_KPCA   import NystromKPCA
from .algorithms.nystrom_KPCR   import NystromKPCR
from .algorithms.nystrom_KRR    import NystromKRR
from .algorithms.kernel_PCR     import KernelPCR
from .algorithms.bound          import calculate_bound, calc_conf_bound
from .experiments import (methods_experiments, bound_experiments,
                          regression_experiments)
