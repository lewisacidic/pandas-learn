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
pdlearn.adaptor.methods
~~~~~~~~~~~~~~~~~~~~~~~

Module implementing methods for pdlearn classes.
"""

import pandas as pd

def feature_property(name):

    """
    Create a method adapting a parent class' property to return a pandas frame.
    """

    # pylint: disable=C0111
    @property
    def method(self):
        # pylint: disable=W0212
        with self._unyouthanize():
            prop = getattr(self, name + '_')
        if self.pandas_mode_:
            return pd.Series(prop, index=self.feature_names_, name=name)
        else:
            return prop
    return method
