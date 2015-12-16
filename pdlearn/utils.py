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
pdlearn.utils
~~~~~~~~~~~~~

Collection of utility functions for the pdlearn package.
"""

import logging
import warnings
from functools import wraps

import pandas as pd


LOGGER = logging.getLogger(__name__)

class CompatabilityWarning(Warning):

    """ Warning to be used when models trained on a different type are used. """
    pass

def is_frame(obj):

    """ Whether an object is considered a dataframe. """

    return isinstance(obj, pd.DataFrame)


def is_series(obj):

    """ Whether an object is considered a series. """

    return isinstance(obj, pd.Series)


def takes_df_or_array(func):

    """
    Decorate a function that takes dataframes to make it also take arrays.
    """

    #pylint: disable=C0111
    @wraps(func)
    def inner(self, X):

        if self.pandas_mode_:
            if not is_frame(X):
                warnings.warn("Trying to use raw array with a model fitted"
                              " with pandas.", CompatabilityWarning)
                X = pd.DataFrame(X, columns=self.feature_names_)
            else:
                X = X[self.feature_names_] # get features in correct order
        return func(self, X)

    return inner


def returns_single_indexed(func):

    """
    Decorate a function to detect whether it should return a dataframe with
    single indexed columns, and if so, to return an appropriate one.

    Expected to be used with takes_df_or_array.
    """

    #pylint: disable=C0111
    @wraps(func)
    def inner(self, X):
        res = func(self, X)

        if self.pandas_mode_:
            if len(self.target_names_) <= 1:
                res = pd.Series(res)
                res.index = X.index
                res.name = self.target_names_[0]
            else:
                res = pd.DataFrame(res)
                res.index = X.index
                res.columns = self.target_names_

        return res

    return inner

def make_multi_index(self):

    """
    Make a multiindex from of target_name-class pairs.
    """
    cls_by_targ = self.classes_ if self.multitask_ else [self.classes_]
    cls_by_targ = zip(self.target_names_, cls_by_targ)
    targ_cls_tup = [(target, klass) \
                for target, classes in cls_by_targ \
                for klass in classes]
    return pd.MultiIndex.from_tuples(targ_cls_tup)

def returns_multi_indexed(func):

    """
    Decorate a function to detect whether it should return a dataframe with
    multi indexed columns, and if so, to return an appropriate one.

    Expected to be used with takes_df_or_array.
    """

    #pylint: disable=C0111
    @wraps(func)
    def inner(self, X):
        res = func(self, X)

        if self.pandas_mode_:

            if self.multitask_:
                res = pd.concat([pd.DataFrame(r) for r in res], axis=1)
            else:
                res = pd.DataFrame(res)

            res.index = X.index

            res.columns = make_multi_index(self)
            return res
        else:
            return res

    return inner
