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
pdlearn.datasets
~~~~~~~~~~~~~~~~

Package adapting datasets from scikit-learn.
"""

from .samples_generator import (
    make_blobs,
    make_classification,
    make_circles,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_gaussian_quantiles,
    make_hastie_10_2,
    make_low_rank_matrix,
    make_moons,
    make_s_curve,
    make_spd_matrix,
    make_swiss_roll,
    make_sparse_coded_signal,
    make_sparse_uncorrelated,
    make_sparse_spd_matrix,
    make_biclusters,
    make_checkerboard,
    make_regression,
    make_multilabel_classification
)

__all__ = [
    "make_blobs",
    "make_classification",
    "make_circles",
    "make_friedman1",
    "make_friedman2",
    "make_friedman3",
    "make_gaussian_quantiles",
    "make_hastie_10_2",
    "make_low_rank_matrix",
    "make_moons",
    "make_s_curve",
    "make_spd_matrix",
    "make_swiss_roll",
    "make_sparse_coded_signal",
    "make_sparse_uncorrelated",
    "make_sparse_spd_matrix",
    "make_biclusters",
    "make_checkerboard",
    "make_regression",
    "make_multilabel_classification"
]
