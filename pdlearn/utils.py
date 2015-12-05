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
pdlearn.utils
~~~~~~~~~~~~~

Collection of utility functions for the pdlearn package.
"""

import logging

import pandas as pd


LOGGER = logging.getLogger(__name__)


def is_frame(obj):

    """ Whether an object is considered a dataframe. """

    return isinstance(obj, pd.DataFrame)


def is_series(obj):

    """ Whether an object is considered a series. """

    return isinstance(obj, pd.Series)
