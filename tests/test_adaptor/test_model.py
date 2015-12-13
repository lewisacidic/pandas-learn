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
tests.test_adaptor.model
~~~~~~~~~~~~~~~~~~~~~~~~

Tests for the model adaptor module of pdlearn.
"""


from pdlearn.adaptor.model import model, fitter

class Parent(object):
    def shout(self):
        return 'parent'
    def fit(self, X, y=None):
        pass

@fitter
def mock_fit(self, X, y=None):
    pass

@model
def model_mock(cls):
    cls.fit = mock_fit
    return cls

@model_mock
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
