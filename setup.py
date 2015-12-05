#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pandas-learn
# https://github.com/RichLewis42/pandas-learn
#
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT
# Copyright (c) 2015, Rich Lewis <rl403@cam.ac.uk>

"""
setup
~~~~~

Setup script for pandas-learn.

"""

from setuptools import setup, find_packages

__author__ = "Rich Lewis"
__copyright__ = "Copyright (c) 2015, Rich Lewis <rl403@cam.ac.uk>"
__credits__ = ["Rich Lewis"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Rich Lewis"
__email__ = "rl403@cam.ac.uk"
__status__ = "Development"

NAME = "pandas-learn"

CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Programming Language :: Python",
]

DESCRIPTION = "Package adapting scikit-learn models to intelligently use "
"pandas data structures."

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
URL = "https://github.com/RichLewis42/pandas-learn"
DOWNLOAD_URL = ""

with open('requirements.txt') as f:
    DEPENDENCIES = [l.strip() for l in f]

with open('requirements_test.txt') as f:
    TEST_DEPENDENCIES = [l.strip() for l in f]

def setup_package():

    """ Run setuptools """

    setup(
        name=NAME,
        maintainer=__maintainer__,
        maintainer_email=__email__,
        description=DESCRIPTION,
        license=__license__,
        url=URL,
        version=__version__,
        download_url=DOWNLOAD_URL,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        install_requires=DEPENDENCIES,
        tests_require=TEST_DEPENDENCIES
    )

if __name__ == "__main__":
    setup_package()
