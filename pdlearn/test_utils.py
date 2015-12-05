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
pdlearn.test_utils
~~~~~~~~~~~~~

Tests for the utils module.
"""

import pdlearn.utils
import pandas as pd
import numpy as np

LIST = [0, 0]

ZERO_D_ARRAY = np.array(0)

ONE_D_ARRAY = np.array([0, 0])

TWO_D_ARRAY = np.array(
    [
        [0, 0],
        [0, 0]
    ])

SERIES = pd.Series([0])

DATAFRAME = pd.DataFrame([0])

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
        assert pdlearn.utils.is_frame(DATAFRAME)

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
        assert not pdlearn.utils.is_series(DATAFRAME)

    def test_series_is(self):
        assert pdlearn.utils.is_series(SERIES)
