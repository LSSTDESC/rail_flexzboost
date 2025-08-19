import numpy as np
import os
import pytest
from rail.core.stage import RailStage
from rail.utils.testing_utils import one_algo
from rail.utils.path_utils import RAILDIR
from rail.core.data import TableHandle
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

def test_flexzboost_skip_grid():
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'trainfrac': 1.0, 'bumpmin': 0.15,
                         'bumpmax': 0.15, 'nbump': 1,
                         'sharpmin': 1.4, 'sharpmax': 1.4,
                         'nsharp': 1, 'max_basis': 35,
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
    results, rerun_results, _ = one_algo("FZBoostskip", train_algo, pz_algo, train_config_dict, estim_config_dict)

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


def test_pq_input_format():
    
    parquetdata = "./tests/validation_10gal.pq"
    train_config_dict = {'zmin': 0.0, 'zmax': 3.0, 'nzbins': 301,
                         'trainfrac': 0.75, 'bumpmin': 0.02,
                         'bumpmax': 0.35, 'nbump': 3,
                         'sharpmin': 0.7, 'sharpmax': 2.1,
                         'nsharp': 3, 'max_basis': 35,
                         'basis_system': 'cosine',
                         'regression_params': {'max_depth': 8,
                                               'objective':
                                               'reg:squarederror'},
                         'hdf5_groupname': '',
                         'model': 'model.tmp'}
    estim_config_dict = {'hdf5_groupname': 'photometry',
                         'model': 'model.tmp',
                         'qp_representation': 'interp'}
    train_algo = flexzboost.FlexZBoostInformer
    pz_algo = flexzboost.FlexZBoostEstimator

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True
    DS.clear()
    
    training_data = DS.read_file('training_data', TableHandle, parquetdata)
    validation_data = DS.read_file('validation_data', TableHandle, parquetdata)

    train_pz = train_algo.make_stage(**train_config_dict)
    train_pz.inform(training_data)

    pz = pz_algo.make_stage(name="FZBoost", **estim_config_dict)
    estim = pz.estimate(validation_data)

    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
