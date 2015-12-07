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
pdlearn.ensemble.forest
~~~~~~~~~~~~~~~~~~~~~~~

Module adapting scikit-learn's forest ensemble models.
"""

from ..adaptor import model, classifier, transformer, feature_property

import sklearn.ensemble

# pylint: disable=C0111
@model
@classifier
class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    feature_importances_ = feature_property('feature_importances')

# disable too many ancestors and abstract not overriden - they are sklearn probs
# pylint: disable=C0111,R0901,W0223
@model
@transformer
class RandomTreesEmbedding(sklearn.ensemble.RandomTreesEmbedding):
    feature_importances_ = feature_property('feature_importances')
