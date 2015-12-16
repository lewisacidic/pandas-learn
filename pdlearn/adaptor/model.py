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
pdlearn.adaptor.model
~~~~~~~~~~~~~~~~~~~~~

Module implementing methods and decorators to add pandas compatability to
classes inheriting from scikit-learn style models.
"""

import contextlib

from ..utils import is_frame, is_series
from functools import wraps

@contextlib.contextmanager
def unyouthanize(self):

    """
    Become old! This context manager will revert an instance to its
    parent's class.

    Used to ensure compatability after changing some public methods that are
    also used privately.
    """

    klass = self.__class__
    self.__class__ = self.__class__.__mro__[1]
    yield
    self.__class__ = klass


@property
def pandas_mode_(self):

    """
    Boolean attribute defining whether a model was trained with pandas data.
    """

    return hasattr(self, 'feature_names_') or hasattr(self, 'target_names_')

def fit_method(func):

    @wraps(func)
    def inner(self, X, y=None):

        func(self, X, y)

        if is_frame(X):
            self.feature_names_ = X.columns

        # pylint: disable=W0212
        with self._unyouthanize():
            self.fit(X, y)

        return self

    return inner


def model(model_decorator):

    """
    Decorator to generically add pandas compatability to classes inheriting from
    a scikit-learn style models.
    """

    @wraps(model_decorator)
    def inner(cls):
        # pylint: disable=W0212
        cls._unyouthanize = unyouthanize
        cls.pandas_mode_ = pandas_mode_

        return model_decorator(cls)

    return inner
