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
pdlearn.adaptor.classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~

Module implementing methods and decorators to add pandas compatability to
classes inheriting from scikit-learn style classifiers.
"""

from ..utils import (
    takes_df_or_array,
    returns_single_indexed,
    returns_multi_indexed)

@property
def multitask_(self):

    """ Whether a model is trained on a multitask problem. """

    # NOTE: this is a bit of a weak check
    return isinstance(self.classes_, list)

# pylint: disable=C0111
@takes_df_or_array
@returns_single_indexed
def predict(self, X):

    # pylint: disable=W0212
    with self._unyouthanize():
        res = self.predict(X)

    return res

# pylint: disable=C0111
@takes_df_or_array
@returns_multi_indexed
def predict_proba(self, X):

    # pylint: disable=W0212
    with self._unyouthanize():
        res = self.predict_proba(X)

    return res

# pylint: disable=C0111
@takes_df_or_array
@returns_multi_indexed
def predict_log_proba(self, X):

    # pylint: disable=W0212
    with self._unyouthanize():
        res = self.predict_log_proba(X)

    return res


def classifier(cls):

    """
    Decorator to generically add pandas compatability to classes inheriting from
    a scikit-learn style classification models.
    """

    cls.predict = predict
    cls.predict_proba = predict_proba
    cls.predict_log_proba = predict_log_proba
    cls.multitask_ = multitask_

    return cls
