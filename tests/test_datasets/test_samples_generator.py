#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of pandas-learn
# https://github.com/RichLewis42/pandas-learn
#
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT
# Copyright (c) 2015, Rich Lewis <rl403@cam.ac.uk>
#
# test module, so disable docstring and method could be function requirements
# pylint: disable=C0111,R0201,E1101

"""
tests.test_datasets.test_samples_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for dataset generator module of pdlearn.
"""

import numpy as np
import pandas as pd
import pytest

# pylint: disable=E0611,no-name-in-module
from pdlearn.datasets.samples_generator import (
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
    make_multilabel_classification,
    SYNTH,
    Synthesizer
)

from ..test_utils import ZERO_D_ARRAY, ONE_D_ARRAY, TWO_D_ARRAY


NORMALS = [
    make_blobs,
    make_classification,
    make_circles,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_gaussian_quantiles,
    make_hastie_10_2,
    make_moons,
    make_s_curve,
    make_swiss_roll,
    make_sparse_uncorrelated
]

FEATS_ONLY = [
    make_low_rank_matrix,
    make_spd_matrix,
    make_sparse_spd_matrix
]

BICLUSTER = [
    make_biclusters,
    make_checkerboard
]


class TestSynthesizer(object):

    def test_random_words(self):
        assert 'arbitrary' in SYNTH.word_list
        assert 'select' in SYNTH.word_list

    def test_random_words_online(self):
        online_synth = Synthesizer(force_dload=True)
        assert 'arbitrary' in online_synth.word_list
        assert 'select' in online_synth.word_list

    def test_words(self):
        assert len(SYNTH.words(10)) == 10

    def test_make_zero_ser(self):
        data = SYNTH.make_pd(ZERO_D_ARRAY)
        assert isinstance(data, pd.Series)
        assert data.index is not None
        assert not np.array_equal(data.index, [0])
        assert data.index.name is not None
        assert data.name is not None

    def test_make_ser(self):
        data = SYNTH.make_pd(ONE_D_ARRAY)
        assert isinstance(data, pd.Series)
        assert data.index is not None
        assert not np.array_equal(data.index, [0, 1])
        assert data.index.name is not None
        assert data.name is not None

    def test_make_df(self):
        data = SYNTH.make_pd(TWO_D_ARRAY)
        assert isinstance(data, pd.DataFrame)
        assert data.index is not None
        assert not np.array_equal(data.index, [0, 1])
        assert data.columns is not None
        assert not np.array_equal(data.columns, [0, 1])
        assert data.index.name is not None
        assert data.columns.name is not None

    def test_make_pds(self):
        data = SYNTH.make_pds(TWO_D_ARRAY, ONE_D_ARRAY)
        assert len(data) == 2
        assert isinstance(data[0], pd.DataFrame)
        assert isinstance(data[1], pd.Series)

    def test_make_pds_same_idx(self):
        data = SYNTH.make_pds(TWO_D_ARRAY, ONE_D_ARRAY, share_index=True)
        assert np.array_equal(data[0].index, data[1].index)

class TestNormalGenerators(object):

    def test_normal_mode(self):
        for gen in NORMALS:
            X, y = gen(pandas_mode=False)
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)

    def test_pandas_mode(self):
        for gen in NORMALS:
            X, y = gen(pandas_mode=True)
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert np.array_equal(X.index, y.index)

class TestRegression(object):

    def test_normal_mode(self):
        X, y = make_regression()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_single_pandas_mode(self):
        X, y = make_regression(pandas_mode=True)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert np.array_equal(X.index, y.index)

    def test_multi_pandas_mode(self):
        X, Y = make_regression(n_targets=5, pandas_mode=True)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(Y, pd.DataFrame)
        assert np.array_equal(X.index, Y.index)

    def test_coef_normal_mode(self):
        X, y, coef = make_regression(coef=True)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(coef, np.ndarray)

    def test_coef_single_pandas_mode(self):
        X, _, coef = make_regression(coef=True, pandas_mode=True)
        assert isinstance(coef, pd.Series)
        assert np.array_equal(X.columns, coef.index)
        assert coef.name == 'coefs'

    def test_coef_multi_pandas_mode(self):
        X, Y, coef = make_regression(n_targets=5, coef=True, pandas_mode=True)
        assert isinstance(coef, pd.DataFrame)
        assert np.array_equal(X.columns, coef.index)
        assert np.array_equal(Y.columns, coef.columns)

class TestMultiLabel(object):

    def test_normal_mode(self):
        X, y = make_multilabel_classification()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_pandas_mode(self):
        X, Y = make_multilabel_classification(n_labels=5, pandas_mode=True)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(Y, pd.DataFrame)
        assert np.array_equal(X.index, Y.index)

    def test_rd_pandas_mode(self):
        X, y, c, w = make_multilabel_classification(n_labels=5,
                                                    return_distributions=True,
                                                    pandas_mode=True)
        assert isinstance(c, pd.Series)
        assert np.array_equal(y.columns, c.index)
        assert c.name == 'prob_of_class'

        assert isinstance(w, pd.DataFrame)
        assert np.array_equal(X.columns, w.index)
        assert np.array_equal(y.columns, w.columns)

    def test_sparse_pandas_mode(self):
        with pytest.raises(NotImplementedError):
            make_multilabel_classification(pandas_mode=True, sparse=True)

class TestBicluster(object):

    def test_normal_mode(self):
        for func in BICLUSTER:
            X, row, col = func(shape=(300, 300), n_clusters=5)
            assert isinstance(X, np.ndarray)
            assert isinstance(row, np.ndarray)
            assert isinstance(col, np.ndarray)

    def test_pandas_mode(self):
        for func in BICLUSTER:
            X, row, col = func(shape=(300, 300), n_clusters=5, pandas_mode=True)
            assert isinstance(X, pd.DataFrame)
            assert isinstance(row, pd.DataFrame)
            assert isinstance(col, pd.DataFrame)


class TestSparseCodedSignal(object):

    def test_normal_mode(self):

        y, X, w = make_sparse_coded_signal(1, 512, 100, 17)
        assert isinstance(y, np.ndarray)
        assert isinstance(X, np.ndarray)
        assert isinstance(w, np.ndarray)

    def test_pandas_mode(self):

        y, X, w = make_sparse_coded_signal(1, 512, 100, 17, pandas_mode=True)
        assert isinstance(y, pd.DataFrame)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(w, pd.DataFrame)
        assert np.array_equal(y.index, X.index)
        assert np.array_equal(y.columns, w.columns)
        assert np.array_equal(X.columns, w.index)
