

from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements_list = f.read().split('\n')


setup(name                 = "nystrompca",
      version              = "1.0.1",
      description          = "Kernel PCA with the Nyström method",
      author               = "Fredrik Hallgren",
      author_email         = "fredrik.hallgren@ucl.ac.uk",
      url                  = "https://github.com/fredhallgren/nystrompca",
      packages             = find_packages(),
      long_description     = """This package implements an efficient non-linear PCA by combining kernel PCA with the Nyström randomized subsampling method, as well as a confidence bound on the accuracy of the method. """,
      long_description_content_type="text/plain",
      classifiers         = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"],
      python_requires      = ">=3.6",
      install_requires     = requirements_list,
      include_package_data = True,
      data_files = [('data', ['nystrompca/data/magic_gamma_telescope.dat',
                              'nystrompca/data/yeast.dat',
                              'nystrompca/data/cardiotocography.dat',
                              'nystrompca/data/segmentation.dat',
                              'nystrompca/data/nips.dat',
                              'nystrompca/data/drug.dat',
                              'nystrompca/data/dailykos.dat',
                              'nystrompca/data/airfoil.dat'])],
      entry_points         = {
          'console_scripts': [
              'nystrompca=nystrompca.experiments.console_script:main']})


