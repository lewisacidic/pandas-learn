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
pdlearn.feature_selection.variance_threshold
~~~~~~~~~~~~~~~~

Package adapting scikit-learn's ensemble models.
"""

import sklearn.feature_selection
from ..adaptor import model, transformer, feature_property

# pylint: disable=C0111
@model
@transformer
class VarianceThreshold(sklearn.feature_selection.VarianceThreshold):

    variances_ = feature_property('variances')

    @property
    def target_names_(self):

        """ Transformed features """

        return self.feature_names_[self.variances_ > self.threshold]
