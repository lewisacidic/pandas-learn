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
pdlearn.linear_model.logistic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module adapting scikit-learn's logistic regresion models.
"""

from ..adaptor import model, classifier, feature_property

import sklearn.linear_model

# pylint: disable=R0901,C0111
@model
@classifier
class LogisticRegression(sklearn.linear_model.LogisticRegression):

    coef_ = feature_property('coef_')
