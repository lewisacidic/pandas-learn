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
pdlearn.adaptor.regresssor
~~~~~~~~~~~~~~~~~~~~~~~~~~

Module implementing adaptor functions for porting scikit-learn's
regressors to work with pandas.
"""

from .classifier import fit, predict
from .model import model

@model
def regressor(cls):

    """
    Decorator to generically add pandas compatability to classes inheriting from
    a scikit-learn style regression models.
    """

    cls.fit = fit
    cls.predict = predict

    return cls
