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
tests.test_adaptor.transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the transformer adaptor module of pdlearn.
"""

import pytest

import pdlearn.adaptor.transformer
import pdlearn.adaptor.model

from ..test_utils import DATA_FRAME, TWO_D_ARRAY
from pdlearn.utils import is_frame, is_series, CompatabilityWarning

import string
import numpy as np

class MockTransformer(object):

    """
    Mock transformer, that 'transforms a feature vector to normally distributed
    random matrix, one dimension smaller than the input dimensionality.
    """

    def transform(self, X, y=None):
        return np.random.randn(X.shape[0], X.shape[1])

@pdlearn.adaptor.transformer
class ChildTransformer(MockTransformer):

    def __init__(self, features=None, targets=None):
        if features:
            self.feature_names_ = features
            default_targs = list(string.ascii_lowercase)[-1:-1-len(features):-1]
            targets = targets if targets else default_targs
        if targets:
            self.target_names_ = targets


class TestTransformer(object):

    """ Tests for pdlearn.adaptor.transformer """

    def test_transform_arr(self):

        res = ChildTransformer().transform(TWO_D_ARRAY)
        assert isinstance(res, np.ndarray)
        assert np.array_equal(res.shape, (2, 2))

    def test_transform_arr_pdmode(self):

        with pytest.warns(CompatabilityWarning):
            res = ChildTransformer(features=['x', 'y']).transform(TWO_D_ARRAY)
        assert is_frame(res)
        assert np.array_equal(res.columns, ['z', 'y'])

    def test_transform_df(self):

        res = ChildTransformer(features=['x', 'y']).transform(DATA_FRAME)
        assert is_frame(res)
        assert np.array_equal(res.columns, ['z', 'y'])
        assert np.array_equal(DATA_FRAME.index, res.index)
