Installation
============

You have many options for installing pandas-learn.

pip
---

pandas-learn is not yet on `PyPI`_.  You are recommended to download it from github with

.. code-block:: bash

  pip install -e git+https://github.com/richlewis42/pandas-learn


setup.py
--------

You and also download the project, then install it using the setup.py file:

.. code-block:: bash

  git clone https://github.com/richlewis42/pandas-learn && cd pandas-learn
  python setup.py install



Conda
-----

The package has a `conda`_ recipe, so can be build using `conda-build`_.

.. code-block:: bash

  git clone https://github.com/richlewis42/pandas-learn && cd pandas-learn
  conda build conda-build



.. _PyPI: pypi.python.org
.. _conda: conda.pydata.org
.. _conda-build: conda.pydata.org/building/recipe
