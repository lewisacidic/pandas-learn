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

# pylint: disable=C0111
@takes_df_or_array
@returns_single_indexed
def transform(self, X):

    # pylint: disable=W0212
    with self._unyouthanize():
        res = self.transform(X)
    return res


def transformer(cls):

    """
    Decorator to generically add pandas compatability to classes inheriting from
    a scikit-learn style transformer models.
    """

    cls.transform = transform
    return cls
