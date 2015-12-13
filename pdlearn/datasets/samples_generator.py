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
pdlearn.datasets.samples_generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module adapting scikit-learn dataset generating capabilities.
"""

from functools import wraps
import os
import random
import sys

import pandas as pd

import sklearn.datasets as ds

# pylint can't seem to find the library, so disable the check
# pylint: disable=E0401,E0611
from sklearn.externals.six.moves.urllib.request import urlopen


class Synthesizer(object):

    """ Helper object for synthesizing dataframes from arrays. """

    online_dict_url = "http://www.lextutor.ca/freq/lists_download/awl_heads.txt"

    def __init__(self, path='/usr/share/dict/words', force_dload=False):
        if os.path.exists(path) and not force_dload: #not on shatter-prone os
            with open(path) as fobj:
                self.word_list = fobj.read().splitlines()
        else:
            words = urlopen(self.online_dict_url).read().splitlines()
            self.word_list = [word.decode().strip().lower() for word in words]

    def word(self):

        """ Return a random word from the dictionary """

        return random.choice(self.word_list)

    def words(self, number=10):

        """ Return a list of words chosen at random from the dictionary """

        return [self.word() for i in range(number)]

    def make_index(self, length=10):

        """ Make an index of a specified length """

        return pd.Index(self.words(length), name=self.word())

    def make_pd(self, arr, index=None):

        """
        Make a dataframe from an array with random words as index and columns.
        Optionally pass in an index to use instead.
        """

        # special case, zero_d array
        if len(arr.shape) == 0:
            idx = self.make_index(1)
            return pd.Series(arr, index=idx, name=self.word())

        # if have to make index
        if index is None:
            index = self.make_index(arr.shape[0])

        if len(arr.shape) == 1:
            return pd.Series(arr,
                             index=index,
                             name=self.word())

        elif len(arr.shape) == 2:
            return pd.DataFrame(arr,
                                index=index,
                                columns=self.make_index(arr.shape[1]))

        else:
            raise NotImplementedError('Can\'t make pandas object from >3tensor')

    def make_pds(self, *args, **kwargs):

        """
        Make dataframes from arrays passed in as arguments, with random
        words as indexes and columns.  Optionally share the index for all arrays
        passed.
        """

        if kwargs.get('share_index'):
            idx = self.make_index(args[0].shape[0])
            return tuple(self.make_pd(a, index=idx) for a in args)
        else:
            return tuple(self.make_pd(a) for a in args)

SYNTH = Synthesizer()

def register_dataset(data_func, name):

    """ Register a dataset by passing its generating function and its name. """

    setattr(sys.modules[__name__], name, data_func)

DATA_GENERATING_FUNCS = (
    ds.make_blobs,
    ds.make_classification,
    ds.make_circles,
    ds.make_friedman1,
    ds.make_friedman2,
    ds.make_friedman3,
    ds.make_gaussian_quantiles,
    ds.make_hastie_10_2,
    ds.make_low_rank_matrix,
    ds.make_moons,
    ds.make_s_curve,
    ds.make_spd_matrix,
    ds.make_swiss_roll,
    ds.make_sparse_uncorrelated,
    ds.make_sparse_spd_matrix
)

def dataset(data_func):

    """
    Decorator for a data set generating function to allow for the pandas_mode_mode
    option, which returns dataframes with synthetic indexes and columns to
    generated data.
    """

    # pylint: disable=C0111
    @wraps(data_func)
    def inner(*args, **kwargs):
        pandas_mode = kwargs.pop('pandas_mode', None)
        res = data_func(*args, **kwargs)
        if pandas_mode:
            return SYNTH.make_pds(*res, share_index=True)
        else:
            return res
    return inner

for func in DATA_GENERATING_FUNCS:
    register_dataset(dataset(func), func.__name__)

# make_regression gets coefs out optionally, so have to handle differently
# pylint: disable=C0111
@wraps(ds.make_regression)
def make_regression(*args, **kwargs):
    if kwargs.get('coef') and kwargs.get('pandas_mode'):
        #handle special case
        kwargs.pop('pandas_mode') # remove like this because py27 is annoying

        X, y, coefs = ds.make_regression(*args, **kwargs)
        X, y = SYNTH.make_pds(X, y, share_index=True)

        if len(coefs.shape) == 1: # single target
            coefs = pd.Series(coefs, index=X.columns, name='coefs')

        else: # multi target
            coefs = pd.DataFrame(coefs, index=X.columns, columns=y.columns)

        return X, y, coefs

    else:
        # it's normal dataset
        return dataset(ds.make_regression)(*args, **kwargs)

# same for make_multilabel_classification and p_c, p_w_c

@wraps(ds.make_multilabel_classification)
def make_multilabel_classification(*args, **kwargs):
    if kwargs.get('sparse') and kwargs.get('pandas_mode'):
        raise NotImplementedError('Sparse datasets are not yet supported in '
                                  'pandas mode.')

    if kwargs.get('return_distributions') and kwargs.get('pandas_mode'):
        kwargs.pop('pandas_mode')
        X, y, p_c, p_w_c = ds.make_multilabel_classification(*args, **kwargs)
        X, y = SYNTH.make_pds(X, y, share_index=True)
        p_c = pd.Series(p_c, index=y.columns, name='prob_of_class')
        p_w_c = pd.DataFrame(p_w_c, index=X.columns, columns=y.columns)
        return X, y, p_c, p_w_c

    else:
        # it's normal dataset
        data_func = dataset(ds.make_multilabel_classification)
        X, y = data_func(*args, **kwargs)
        return X, y

# unique dataset generator needs own implementation
@wraps(ds.make_sparse_coded_signal)
def make_sparse_coded_signal(n_samples, n_components, n_features,
                             n_nonzero_coefs, random_state=None,
                             pandas_mode=False):

    y, X, w = ds.make_sparse_coded_signal(n_samples, n_components, n_features,
                                 n_nonzero_coefs, random_state=None)
    if not pandas_mode:
        return y, X, w
    else:
        comps = pd.Index(['comp-{}'.format(i) for i in range(n_components)],
                         name='components')
        feats = pd.Index(['feat-{}'.format(i) for i in range(n_features)],
                         name='features')
        samps = pd.Index(['sample-{}'.format(i) for i in range(n_samples)],
                         name='samples')

        y = pd.DataFrame(y, index=feats, columns=samps)
        X = pd.DataFrame(X, index=feats, columns=comps)
        w = pd.DataFrame(w, index=comps, columns=samps)
        return y, X, w


# bicluster datasets return rows and columns, so also need special treatment
def bicluster_dataset(data_func):

    """
    Decorator for biclusting dataset generating functions to add pandas_mode
    functionality
    """

    @wraps(data_func)
    def inner(*args, **kwargs):
        pandas_mode = kwargs.pop('pandas_mode', None)
        X, rows, cols = data_func(*args, **kwargs)
        if pandas_mode:
            X = SYNTH.make_pd(X)
            rows = pd.DataFrame(rows, columns=X.index)
            cols = pd.DataFrame(cols, columns=X.columns)
            rows.columns.name = cols.columns.name = 'cluster_idx'
        return X, rows, cols
    return inner

BICLUSTER_DATA_GENERATING_FUNCS = (
    ds.make_biclusters,
    ds.make_checkerboard
)

for func in BICLUSTER_DATA_GENERATING_FUNCS:
    register_dataset(bicluster_dataset(func), func.__name__)
