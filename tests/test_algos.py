import numpy as np
import pytest
from rail.core.stage import RailStage
from rail.utils.testing_utils import one_algo
from rail.utils.path_utils import RAILDIR
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
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp',
                         'calculated_point_estimates': ['mode', 'mean']}

    train_algo = flexzboost.FlexZBoostInformer
    pz_algo = flexzboost.FlexZBoostEstimator
    results, rerun_results, _ = one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)

    assert np.isclose(results.ancil['mode'], rerun_results.ancil['mode']).all()
    assert np.isclose(results.ancil['mean'], rerun_results.ancil['mean']).all()

def test_flexzboost_with_interp():
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
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp',
                         'qp_representation': 'interp',
                         'calculated_point_estimates': ['mode', 'mean', 'median']}

    train_algo = flexzboost.FlexZBoostInformer
    pz_algo = flexzboost.FlexZBoostEstimator
    results, rerun_results, _ = one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)

    assert np.isclose(results.ancil['mode'], rerun_results.ancil['mode']).all()
    assert np.isclose(results.ancil['mean'], rerun_results.ancil['mean']).all()
    assert np.isclose(results.ancil['median'], rerun_results.ancil['median']).all()


@pytest.mark.slow
def test_flexzboost_with_qp_flexzboost():
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
    estim_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'hdf5_groupname': 'photometry',
                         'model': 'model.tmp',
                         'qp_representation': 'flexzboost',
                         'calculated_point_estimates': ['mode', 'mean', 'median']}

    train_algo = flexzboost.FlexZBoostInformer
    pz_algo = flexzboost.FlexZBoostEstimator
    results, rerun_results, _ = one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)

    assert np.isclose(results.ancil['mode'], rerun_results.ancil['mode']).all()
    assert np.isclose(results.ancil['mean'], rerun_results.ancil['mean']).all()
    assert np.isclose(results.ancil['median'], rerun_results.ancil['median']).all()


def test_flexzboost_with_unknown_qp_representation():
    """Pass a bogus qp_representation string to the config, expect a ValueError"""
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
                         'model': 'model.tmp',
                         'qp_representation': 'bogus'}

    train_algo = flexzboost.FlexZBoostInformer
    pz_algo = flexzboost.FlexZBoostEstimator
    with pytest.raises(ValueError) as excinfo:
        one_algo("FZBoost", train_algo, pz_algo, train_config_dict, estim_config_dict)
        assert "Unknown qp_representation" in str(excinfo.value)

def test_catch_bad_bands():
    params = dict(bands='u,g,r,i,z,y')
    with pytest.raises(ValueError):
        flexzboost.FlexZBoostInformer.make_stage(hdf5_groupname='', **params)
    with pytest.raises(ValueError):
        flexzboost.FlexZBoostEstimator.make_stage(hdf5_groupname='', **params)

def test_missing_groupname_keyword():
    """hdf5_groupname will default to 'photometry'."""

    config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                   'trainfrac': 0.75, 'bumpmin': 0.02,
                   'bumpmax': 0.35, 'nbump': 3,
                   'sharpmin': 0.7, 'sharpmax': 2.1,
                   'nsharp': 3, 'max_basis': 35,
                   'basis_system': 'cosine',
                   'regression_params': {'max_depth': 8,
                                             'objective':
                                             'reg:squarederror'}}
    stage = flexzboost.FlexZBoostEstimator.make_stage(**config_dict)
    assert stage.config_options['hdf5_groupname'] == "photometry"
