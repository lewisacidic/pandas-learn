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
tests.test_adaptor.regressor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the regressor adaptor module of pdlearn.
"""

import pytest

from ..test_utils import DATA_FRAME, TWO_D_ARRAY
from pdlearn.utils import is_frame, is_series, CompatabilityWarning

import pdlearn.adaptor.regressor
import pdlearn.adaptor.model
import numpy as np
import pandas as pd


class MockRegressor(object):

    def __init__(self, multitask=False):
        if multitask:
            self.classes_ = [np.array([0, 1]), np.array([0, 1])]
        else:
            self.classes_ = np.array([0, 1])

        self._num_tasks = 2 if multitask else 1

    def predict(self, X):
        if self._num_tasks == 1:
            return np.array([42] * len(X))
        else:
            return [np.array([42] * len(X))] * self._num_tasks

@pdlearn.adaptor.regressor
class ChildRegressor(MockRegressor):

    def __init__(self, features=None, targets=None, multitask=False):

        if targets:
            self.target_names_ = pd.Index(targets)
            multitask = multitask or len(self.target_names_) > 1
        if features:
            self.feature_names_ = pd.Index(features)

        super(ChildRegressor, self).__init__(multitask)


class TestRegressor(object):

    """ Tests for pdlearn.adaptor.regressor """

    def test_predict(self):

        res = ChildRegressor().predict(TWO_D_ARRAY)
        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, [42, 42])

    def test_predict_arr_pdmode(self):

        c = ChildRegressor(features=['x', 'y'], targets=['a'])

        with pytest.warns(CompatabilityWarning):
            res = c.predict(TWO_D_ARRAY)

        assert is_series(res)
        assert np.array_equal(res.index, range(len(TWO_D_ARRAY)))
        assert res.name == 'a'

    def test_predict_df(self):

        c = ChildRegressor(features=['x', 'y'], targets=['c'])

        res = c.predict(DATA_FRAME)
        assert is_series(res)
        assert np.array_equal(res.index, DATA_FRAME.index)
        assert res.name == 'c'

    def test_predict_df_multitask(self):

        c = ChildRegressor(features=['x', 'y'], targets=['c', 'd'])

        res = c.predict(DATA_FRAME)
        assert is_frame(res)
        assert np.array_equal(res.columns, ['c', 'd'])
        assert np.array_equal(res.index, DATA_FRAME.index)
    pass
