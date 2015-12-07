#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pandas-learn
# https://github.com/RichLewis42/pandas-learn
#
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT
# Copyright (c) 2015, Rich Lewis <rl403@cam.ac.uk>
# pylint: disable=C0111

"""
pdlearn.feature_selection.tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the feature selection package of pdlearn.
"""

import pandas as pd

from pdlearn.feature_selection import VarianceThreshold
from pdlearn.utils import is_frame, is_series


DATA = pd.DataFrame(
    [
        [4, 6, 9],
        [4, 7, 8],
        [4, 3, 6]
    ],
    index=['a', 'b', 'c'],
    columns=['zero_var', 'non_zero_1', 'non_zero_2'])

def test_variances_are_dfs():
    v_t = VarianceThreshold(0)
    v_t.fit(DATA)
    assert is_series(v_t.variances_)

def test_removes_novariance_column():
    v_t = VarianceThreshold(0)
    res_df = v_t.fit_transform(DATA)
    assert is_frame(res_df)
    assert 'zero_var' not in res_df.columns

def test_works_with_reordered_feats():
    v_t = VarianceThreshold(0)
    v_t.fit(DATA)
    res_df = v_t.transform(DATA[['non_zero_1', 'zero_var', 'non_zero_2']])
    assert 'zero_var' not in res_df.columns
