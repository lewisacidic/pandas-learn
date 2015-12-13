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
pdlearn.adaptor
~~~~~~~~~~~~~~~

Package of adaptor methods and functions to simplify porting scikit-learn's
models to work transparently with pandas.
"""

from .classifier import classifier
from .regressor import regressor
from .transformer import transformer
from .methods import feature_property
