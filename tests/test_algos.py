import numpy as np
import os
import sys
import glob
import pickle
import pytest
import yaml
import tables_io
from rail.core.stage import RailStage
from rail.core.data import DataStore, TableHandle
from rail.core.algo_utils import one_algo
from rail.core.utils import RAILDIR
from rail.estimation.algos import flexzboost
import scipy.special
sci_ver_str = scipy.__version__.split('.')


def test_flexzboost():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'trainfrac': 0.75, 'bumpmin': 0.02,
                         'bumpmax': 0.35, 'nbump': 3,
                         'sharpmin': 0.7, 'sharpmax': 2.1,
                         'nsharp': 3, 'max_basis': 35,
                         'basis_system': 'cosine',
                         'regression_params': {'max_depth': 8,
                                               'objective':
                                               'reg:squarederror'},
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    estim_config_dict = {'hdf5_groupname': 'photometry',
                         'model': 'model.tmp'}
    # zb_expected = np.array([0.13, 0.13, 0.13, 0.12, 0.12, 0.13, 0.12, 0.13,
    #                         0.12, 0.12])
    train_algo = flexzboost.Inform_FZBoost
    pz_algo = flexzboost.FZBoost
    results, rerun_results, rerun3_results = one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)
    # assert np.isclose(results.ancil['zmode'], zb_expected).all()
    assert np.isclose(results.ancil['zmode'], rerun_results.ancil['zmode']).all()


def test_catch_bad_bands():
    params = dict(bands='u,g,r,i,z,y')
    with pytest.raises(ValueError):
        flexzboost.Inform_FZBoost.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError):
        flexzboost.FZBoost.make_stage(hdf5_groupname='', **params)


def test_missing_groupname_keyword():
    config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                   'trainfrac': 0.75, 'bumpmin': 0.02,
                   'bumpmax': 0.35, 'nbump': 3,
                   'sharpmin': 0.7, 'sharpmax': 2.1,
                   'nsharp': 3, 'max_basis': 35,
                   'basis_system': 'cosine',
                   'regression_params': {'max_depth': 8,
                                             'objective':
                                             'reg:squarederror'}}
    with pytest.raises(ValueError):
        _ = flexzboost.FZBoost.make_stage(**config_dict)


def test_wrong_modelfile_keyword():
    RailStage.data_store.clear()
    config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                   'trainfrac': 0.75, 'bumpmin': 0.02,
                   'bumpmax': 0.35, 'nbump': 3,
                   'sharpmin': 0.7, 'sharpmax': 2.1,
                   'nsharp': 3, 'max_basis': 35,
                   'basis_system': 'cosine',
                   'hdf5_groupname': 'photometry',
                   'regression_params': {'max_depth': 8,
                                             'objective':
                                             'reg:squarederror'},
                   'model': 'nonexist.pkl'}
    with pytest.raises(FileNotFoundError):
        pz_algo = flexzboost.FZBoost.make_stage(**config_dict)
        assert pz_algo.model is None
