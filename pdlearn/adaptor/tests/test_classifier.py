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
pdlearn.adaptor.tests.classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the classifier adaptor module.
"""

import pytest

from pdlearn.test_utils import DATA_FRAME, TWO_D_ARRAY
from pdlearn.utils import is_frame, is_series, CompatabilityWarning

import pdlearn.adaptor.classifier
import pdlearn.adaptor.model
import numpy as np
import pandas as pd


class MockClassifier(object):

    def __init__(self, multitask=False):
        if multitask:
            self.classes_ = [np.array([0, 1]), np.array([0, 1])]
        else:
            self.classes_ = np.array([0, 1])

        self._num_tasks = 2 if multitask else 1

    def predict(self, X):
        if self._num_tasks == 1:
            return np.array([1] * len(X))
        else:
            return [np.array([1] * len(X))] * self._num_tasks

    def predict_proba(self, X):

        if self._num_tasks == 1:
            return np.array([[0.1, 0.9]] * len(X))
        else:
            return [np.array([[0.1, 0.9]] * len(X))] * self._num_tasks

    def predict_log_proba(self, X):
        if self._num_tasks == 1:
            return np.array([[-1, -0.04]] * len(X))
        else:
            return [np.array([[-1, -0.04]] * len(X))] * self._num_tasks


@pdlearn.adaptor.model
@pdlearn.adaptor.classifier
class ChildClassifier(MockClassifier):

    def __init__(self, features=None, targets=None, multitask=False):

        if targets:
            self.target_names_ = pd.Index(targets)
            multitask = multitask or len(self.target_names_) > 1
        if features:
            self.feature_names_ = pd.Index(features)

        super(ChildClassifier, self).__init__(multitask)

class TestClassifier(object):

    """ Tests for pdlearn.adaptor.classifier """

    def test_not_multitask(self):

        assert not ChildClassifier(targets=['a']).multitask_

    def test_multitask(self):

        assert ChildClassifier(targets=['a', 'b']).multitask_

    def test_predict(self):

        res = ChildClassifier().predict(TWO_D_ARRAY)
        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, [1, 1])

    def test_predict_arr_pdmode(self):

        c = ChildClassifier(features=['x', 'y'], targets=['a'])

        with pytest.warns(CompatabilityWarning):
            res = c.predict(TWO_D_ARRAY)

        assert is_series(res)
        assert np.array_equal(res.index, range(len(TWO_D_ARRAY)))
        assert res.name == 'a'

    def test_predict_df(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c'])

        res = c.predict(DATA_FRAME)
        assert is_series(res)
        assert np.array_equal(res.index, DATA_FRAME.index)
        assert res.name == 'c'

    def test_predict_df_multitask(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c', 'd'])

        res = c.predict(DATA_FRAME)
        assert is_frame(res)
        assert np.array_equal(res.columns, ['c', 'd'])
        assert np.array_equal(res.index, DATA_FRAME.index)

    def test_predict_proba_arr(self):

        c = ChildClassifier()
        res = c.predict(TWO_D_ARRAY)
        assert isinstance(res, np.ndarray)

    def test_predict_proba_arr_pdmode(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c'])

        with pytest.warns(CompatabilityWarning):
            res = c.predict_proba(TWO_D_ARRAY)
        assert is_frame(res)
        assert np.array_equal(res.index, range(len(TWO_D_ARRAY)))
        assert isinstance(res.columns, pd.MultiIndex)
        assert np.array_equal(res.columns.get_level_values(0), ['c', 'c'])
        assert np.array_equal(res.columns.get_level_values(1), [0, 1])

    def test_predict_proba_df(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c'])
        res = c.predict_proba(DATA_FRAME)
        assert is_frame(res)
        assert np.array_equal(res.index, DATA_FRAME.index)
        assert isinstance(res.columns, pd.MultiIndex)
        assert np.array_equal(res.columns.get_level_values(0), ['c', 'c'])
        assert np.array_equal(res.columns.get_level_values(1), [0, 1])

    def test_predict_proba_multitask(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c', 'd'])
        res = c.predict_proba(DATA_FRAME)
        assert np.array_equal(res.index, DATA_FRAME.index)
        assert isinstance(res.columns, pd.MultiIndex)
        assert np.array_equal(res.columns.get_level_values(0), ['c', 'c', 'd', 'd'])
        assert np.array_equal(res.columns.get_level_values(1), [0, 1, 0, 1])

    def test_predict_log_proba_arr(self):

        res = ChildClassifier().predict_log_proba(TWO_D_ARRAY)
        assert isinstance(res, np.ndarray)

    def test_predict_log_proba_arr_pdmode(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c'])
        with pytest.warns(CompatabilityWarning):
            res = c.predict_log_proba(TWO_D_ARRAY)
        assert is_frame(res)
        assert np.array_equal(res.index, range(len(TWO_D_ARRAY)))
        assert isinstance(res.columns, pd.MultiIndex)
        assert np.array_equal(res.columns.get_level_values(0), ['c', 'c'])
        assert np.array_equal(res.columns.get_level_values(1), [0, 1])

    def test_predict_log_proba_df(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c'])
        res = c.predict_log_proba(DATA_FRAME)
        assert is_frame(res)
        assert np.array_equal(res.index, DATA_FRAME.index)
        assert isinstance(res.columns, pd.MultiIndex)
        assert np.array_equal(res.columns.get_level_values(0), ['c', 'c'])
        assert np.array_equal(res.columns.get_level_values(1), [0, 1])

    def test_predict_log_proba_multitask(self):

        c = ChildClassifier(features=['x', 'y'], targets=['c', 'd'])
        res = c.predict_log_proba(DATA_FRAME)
        assert np.array_equal(res.index, DATA_FRAME.index)
        assert isinstance(res.columns, pd.MultiIndex)
        assert np.array_equal(res.columns.get_level_values(0), ['c', 'c', 'd', 'd'])
        assert np.array_equal(res.columns.get_level_values(1), [0, 1, 0, 1])
