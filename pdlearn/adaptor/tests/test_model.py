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
pdlearn.adaptor.tests.model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the model adaptor module.
"""


import pdlearn.adaptor
from pdlearn.test_utils import DATA_FRAME, SERIES, ONE_D_ARRAY, TWO_D_ARRAY

import numpy as np
import pandas as pd

TARGETS = pd.DataFrame(TWO_D_ARRAY, columns=['T', 'G'])

class Parent(object):
    def shout(self):
        return 'parent'
    def fit(self, X, y=None):
        pass


@pdlearn.adaptor.model
class Child(Parent):

    def __init__(self, feature_names=None, target_names=None):
        if feature_names:
            self.feature_names_ = feature_names
        if target_names:
            self.target_names_ = target_names

    def shout(self):
        return 'child'


class TestModel(object):

    """ Tests for pdlearn.adaptor.model """

    def test_unyouthanize(self):

        child = Child()

        assert child.shout() == 'child'
        with child._unyouthanize():
            assert child.shout() == 'parent'
        assert child.shout() == 'child'

    def test_pandas_mode(self):

        assert not Child().pandas_mode_
        assert Child(feature_names=['a', 'b']).pandas_mode_
        assert Child(target_names=['c', 'd']).pandas_mode_
        assert Child(feature_names=['a', 'b'],
                     target_names=['c', 'd']).pandas_mode_

    def test_fit_with_arr(self):

        child = Child().fit(TWO_D_ARRAY)
        assert not child.pandas_mode_

        child.fit(TWO_D_ARRAY, ONE_D_ARRAY)
        assert not child.pandas_mode_

    def test_with_hybrid(self):

        child = Child().fit(TWO_D_ARRAY, SERIES)
        assert child.pandas_mode_
        assert np.array_equal(child.target_names_, ['T'])

        child.fit(DATA_FRAME, ONE_D_ARRAY)
        assert child.pandas_mode_
        assert np.array_equal(child.feature_names_, ['x', 'y'])

    def test_fit_with_df(self):

        child = Child().fit(DATA_FRAME)
        assert child.pandas_mode_
        assert np.array_equal(child.feature_names_, ['x', 'y'])

    def test_fit_with_df_ser(self):

        child = Child().fit(DATA_FRAME, SERIES)
        assert child.pandas_mode_
        assert np.array_equal(child.feature_names_, ['x', 'y'])
        assert np.array_equal(child.target_names_, ['T'])

    def test_fit_multitarget(self):

        child = Child().fit(DATA_FRAME, TARGETS)
        assert child.pandas_mode_
        assert np.array_equal(child.feature_names_, ['x', 'y'])
        assert np.array_equal(child.target_names_, ['T', 'G'])
