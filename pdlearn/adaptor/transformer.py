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
pdlearn.adaptor.transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module implementing adaptor functions for porting scikit-learn's
transformers to work with pandas.
"""

from ..utils import takes_df_or_array, returns_single_indexed
from .model import model, fit_method

# pylint: disable=C0111
@takes_df_or_array
@returns_single_indexed
def transform(self, X):

    # pylint: disable=W0212
    with self._unyouthanize():
        res = self.transform(X)
    return res

# Transformers do not know what their targets are ahead of time, as that is
# determined during the fit.
#
# pylint: disable=C0111
@fit_method
def fit(self, X, y=None):
    pass


@model
def transformer(cls):

    """
    Decorator to generically add pandas compatability to classes inheriting from
    a scikit-learn style transformer models.
    """

    cls.fit = fit
    cls.transform = transform
    return cls
