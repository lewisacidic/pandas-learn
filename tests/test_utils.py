#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pandas-learn
# https://github.com/RichLewis42/pandas-learn
#
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT
# Copyright (c) 2015, Rich Lewis <rl403@cam.ac.uk>
# pylint: disable=missing-docstring,R0201
"""
test.test_utils
~~~~~~~~~~~~~~~

Tests for the utils module of pdlearn.
"""

import pytest

import pdlearn.utils
import pandas as pd
import numpy as np

LIST = [1, 2]

ZERO_D_ARRAY = np.array(0)

ONE_D_ARRAY = np.array([1, 2])

TWO_D_ARRAY = np.array(
    [
        [1, 2],
        [3, 4]
    ])

SERIES = pd.Series(ONE_D_ARRAY, index=['a', 'b'], name='T')

DATA_FRAME = pd.DataFrame(TWO_D_ARRAY, index=['a', 'b'], columns=['x', 'y'])

ARRAY_MEAN = np.array([1.5, 3.5])


class TestIsFrame(object):

    """
    Test pdlearn.utils.is_frame
    """

    def test_list_isnt(self):
        assert not pdlearn.utils.is_frame(LIST)

    def test_zero_d_array_isnt(self):
        assert not pdlearn.utils.is_frame(ZERO_D_ARRAY)

    def test_one_d_array_isnt(self):
        assert not pdlearn.utils.is_frame(ONE_D_ARRAY)

    def test_two_d_array_isnt(self):
        assert not pdlearn.utils.is_frame(TWO_D_ARRAY)

    def test_series_isnt(self):
        assert not pdlearn.utils.is_frame(SERIES)

    def test_dataframe_is(self):
        assert pdlearn.utils.is_frame(DATA_FRAME)

class TestIsSeries(object):

    """
    Test pdlearn.utils.is_series
    """

    def test_list_isnt(self):
        assert not pdlearn.utils.is_series(LIST)

    def test_zero_d_array_isnt(self):
        assert not pdlearn.utils.is_series(ZERO_D_ARRAY)

    def test_one_d_array_isnt(self):
        assert not pdlearn.utils.is_series(ONE_D_ARRAY)

    def test_two_d_array_isnt(self):
        assert not pdlearn.utils.is_series(TWO_D_ARRAY)

    def test_dataframe_isnt(self):
        assert not pdlearn.utils.is_series(DATA_FRAME)

    def test_series_is(self):
        assert pdlearn.utils.is_series(SERIES)

class Mocked(object):

    def __init__(self, feature_names=None, target_names=None):
        self.pandas_mode_ = not feature_names == target_names == None
        if feature_names:
            self.feature_names_ = pd.Index(feature_names)
        if target_names:
            self.target_names_ = pd.Index(target_names)

    @pdlearn.utils.takes_df_or_array
    def multiply_by_two(self, arr):
        arr = arr.copy()
        arr *= 2
        return arr

    @pdlearn.utils.returns_single_indexed
    def mean(self, arr):
        return np.mean(arr, axis=1)

    @pdlearn.utils.returns_single_indexed
    def identity(self, arr):
        return arr.copy()

RES_ARRAY = np.array(
    [
        [2, 4],
        [6, 8]
    ])

class TestTakesDfOrArray(object):

    """
    Test pdlearn.utils.takes_df_or_array
    """

    def test_with_arr_no_pdmode(self):

        mocked = Mocked(feature_names=None)
        res = mocked.multiply_by_two(TWO_D_ARRAY)
        assert np.array_equal(res, RES_ARRAY)
        assert isinstance(res, np.ndarray)

    def test_warns_with_arr_in_pdmode(self):

        mocked = Mocked(feature_names=['x', 'y'])
        with pytest.warns(pdlearn.utils.CompatabilityWarning):
            res = mocked.multiply_by_two(TWO_D_ARRAY)
        assert np.array_equal(res, RES_ARRAY)
        assert pdlearn.utils.is_frame(res)

    def test_with_df_no_pdmode(self):

        mocked = Mocked(feature_names=None)
        res = mocked.multiply_by_two(DATA_FRAME)
        assert np.array_equal(res, RES_ARRAY)
        assert pdlearn.utils.is_frame(res)

    def test_with_df_pdmode(self):

        mocked = Mocked(feature_names=['x', 'y'])
        print(mocked.feature_names_)
        print(DATA_FRAME)
        print(DATA_FRAME[mocked.feature_names_])
        res = mocked.multiply_by_two(DATA_FRAME)
        assert np.array_equal(res, RES_ARRAY)
        assert pdlearn.utils.is_frame(res)

    def test_feats_order_swap(self):

        mocked = Mocked(feature_names=['y', 'x'])
        res = mocked.multiply_by_two(DATA_FRAME)
        assert np.array_equal(res, np.array([[4, 2], [8, 6]]))
        assert pdlearn.utils.is_frame(res)


class TestReturnsSingleIndexed(object):

    """
    Test pdlearn.utils.returns_single_indexed
    """

    def test_return_series_with_no_pdmode(self):

        mocked = Mocked(target_names=None)
        res = mocked.mean(TWO_D_ARRAY)
        assert np.array_equal(res, ARRAY_MEAN)
        assert isinstance(res, np.ndarray)

    def test_return_series_with_pdmode(self):

        mocked = Mocked(target_names=['Z'])
        res = mocked.mean(DATA_FRAME)
        assert np.array_equal(res, ARRAY_MEAN)
        assert pdlearn.utils.is_series(res)
        assert res.name == 'Z'

    def test_return_df_with_no_pdmode(self):

        mocked = Mocked(target_names=None)
        res = mocked.identity(TWO_D_ARRAY)
        assert np.array_equal(res, TWO_D_ARRAY)
        assert isinstance(res, np.ndarray)

    def test_return_df_with_pdmode(self):

        mocked = Mocked(target_names=['Z', 'B'])
        res = mocked.identity(DATA_FRAME)
        print(res)
        assert np.array_equal(res, TWO_D_ARRAY)
        assert pdlearn.utils.is_frame(res)
        assert np.array_equal(res.columns, ['Z', 'B'])


class TestReturnsMultiIndexed(object):

    """
    Test pdlearn.utils.returns_multi_indexed
    """

    # TODO: Implement these tests.
    pass
