"""
Implementation of the FlexZBoost algorithm, uses training data and
XGBoost to learn the relation, split training data into train and
validation set and find best "bump_thresh" (eliminate small peaks in
p(z) below threshold) and sharpening parameter (determines peakiness of
p(z) shape) via cde-loss over a grid.
"""

import numpy as np
import pandas as pd
import qp
import qp_flexzboost
from ceci.config import StageParameter as Param
from flexcode.helpers import make_grid
from rail.estimation.estimator import CatEstimator, CatInformer
from rail.core.common_params import SHARED_PARAMS


def make_color_data(data_dict, bands, err_bands, ref_band):
    """
    make a dataset consisting of the i-band mag and the five colors.

    Parameters
    -----------
    data_dict : `ndarray`
      array of magnitudes and errors, with names mag_{bands[i]}_lsst
      and mag_err_{bands[i]}_lsst respectively.

    Returns
    --------
    input_data : `ndarray`
      array of imag and 5 colors
    """
    input_data = data_dict[ref_band]
    nbands = len(bands) - 1
    for i in range(nbands):
        color = data_dict[bands[i]] - data_dict[bands[i + 1]]
        input_data = np.vstack((input_data, color))
        colorerr = np.sqrt(data_dict[err_bands[i]]**2 + data_dict[err_bands[i + 1]]**2)
        np.vstack((input_data, colorerr))
    return input_data.T


class FlexZBoostInformer(CatInformer):
    """ Train a FlexZBoost CatInformer
    """
    name = 'FlexZBoostInformer'
    config_options = CatInformer.config_options.copy()
    config_options.update(zmin=SHARED_PARAMS,
                          zmax=SHARED_PARAMS,
                          nzbins=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          err_bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          retrain_full=Param(bool, True, msg="if True, re-run the fit with the full training set, "
                                             "including data set aside for bump/sharpen validation.  If False, only"
                                             " use the subset defined via trainfrac fraction"),
                          trainfrac=Param(float, 0.75,
                                          msg="fraction of training "
                                          "data to use for training (rest used for bump thresh "
                                          "and sharpening determination)"),
                          seed=Param(int, 1138, msg="Random number seed"),
                          bumpmin=Param(float, 0.02,
                                        msg="minimum value in grid of "
                                        "thresholds checked to optimize removal of spurious "
                                        "small bumps"),
                          bumpmax=Param(float, 0.35,
                                        msg="max value in grid checked "
                                            "for removal of small bumps"),
                          nbump=Param(int, 20, msg="number of grid points in bumpthresh grid search"),
                          sharpmin=Param(float, 0.7, msg="min value in grid checked in optimal sharpening parameter fit"),
                          sharpmax=Param(float, 2.1, msg="max value in grid checked in optimal sharpening parameter fit"),
                          nsharp=Param(int, 15, msg="number of search points in sharpening fit"),
                          max_basis=Param(int, 35, msg="maximum number of basis funcitons to use in density estimate"),
                          basis_system=Param(str, 'cosine', msg="type of basis sytem to use with flexcode"),
                          regression_params=Param(dict, {'max_depth': 8, 'objective': 'reg:squarederror'},
                                                  msg="dictionary of options passed to flexcode, includes "
                                                  "max_depth (int), and objective, which should be set "
                                                  " to reg:squarederror"))

    def __init__(self, args, comm=None):
        """ Constructor
        Do CatInformer specific initialization, then check on bands """
        CatInformer.__init__(self, args, comm=comm)
        if self.config.ref_band not in self.config.bands:
            raise ValueError("ref_band not present in bands list! ")

    @staticmethod
    def split_data(fz_data, sz_data, trainfrac, seed):
        """
        make a random partition of the training data into training and
        validation, validation data will be used to determine bump
        thresh and sharpen parameters.
        """
        nobs = fz_data.shape[0]
        ntrain = round(nobs * trainfrac)
        # set a specific seed for reproducibility
        rng = np.random.default_rng(seed=seed)
        perm = rng.permutation(nobs)
        x_train = fz_data[perm[:ntrain], :]
        z_train = sz_data[perm[:ntrain]]
        x_val = fz_data[perm[ntrain:]]
        z_val = sz_data[perm[ntrain:]]
        return x_train, x_val, z_train, z_val

    def run(self):
        """Train flexzboost model model
        """
        import flexcode
        from flexcode.regression_models import XGBoost
        from flexcode.loss_functions import cde_loss

        if self.config.hdf5_groupname:
            training_data = self.get_data('input')[self.config.hdf5_groupname]
        else:  # pragma: no cover
            training_data = self.get_data('input')
        speczs = np.array(training_data[self.config['redshift_col']])

        # replace nondetects
        for bandname, errname in zip(self.config.bands, self.config.err_bands):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                detmask = np.isnan(training_data[bandname])
                if isinstance(training_data, pd.DataFrame):
                    training_data.loc[detmask, bandname] = self.config.mag_limits[bandname]
                    training_data.loc[detmask, errname] = 1.0
                else:
                    detmask = np.isnan(training_data[bandname])
                    training_data[bandname][detmask] = self.config.mag_limits[bandname]
                    training_data[errname][detmask] = 1.0
            else:
                detmask = np.isclose(training_data[bandname], self.config.nondetect_val, atol=0.01)
                if isinstance(training_data, pd.DataFrame):  # pragma: no cover
                    training_data.loc[detmask, bandname] = self.config.mag_limits[bandname]
                    training_data.loc[detmask, errname] = 1.0
                else:
                    training_data[bandname][detmask] = self.config.mag_limits[bandname]
                    training_data[errname][detmask] = 1.0

        print("stacking some data...")
        color_data = make_color_data(training_data, self.config.bands, self.config.err_bands,
                                     self.config.ref_band)
        train_dat, val_dat, train_sz, val_sz = self.split_data(color_data,
                                                               speczs,
                                                               self.config.trainfrac,
                                                               self.config.seed)
        print("read in training data")
        model = flexcode.FlexCodeModel(XGBoost, max_basis=self.config.max_basis,
                                       basis_system=self.config.basis_system,
                                       z_min=self.config.zmin, z_max=self.config.zmax,
                                       regression_params=self.config.regression_params)
        print("fit the model...")
        model.fit(train_dat, train_sz)
        bump_grid = np.linspace(self.config.bumpmin, self.config.bumpmax, self.config.nbump)
        print("finding best bump thresh...")
        bestloss = 9999
        for bumpt in bump_grid:
            model.bump_threshold = bumpt
            model.tune(val_dat, val_sz)
            tmpcdes, z_grid = model.predict(val_dat, n_grid=self.config.nzbins)
            tmploss = cde_loss(tmpcdes, z_grid, val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestbump = bumpt
        model.bump_threshold = bestbump
        print("finding best sharpen parameter...")
        sharpen_grid = np.linspace(self.config.sharpmin, self.config.sharpmax, self.config.nsharp)
        bestloss = 9999
        bestsharp = 9999
        for sharp in sharpen_grid:
            model.sharpen_alpha = sharp
            tmpcdes, z_grid = model.predict(val_dat, n_grid=301)
            tmploss = cde_loss(tmpcdes, z_grid, val_sz)
            if tmploss < bestloss:
                bestloss = tmploss
                bestsharp = sharp
        model.sharpen_alpha = bestsharp

        # retrain with full dataset or not
        if self.config.retrain_full:
            print("Retraining with full training set...")
            model.fit(color_data, speczs)
        else:  # pragma: no cover
            print(f"Skipping retraining, only fraction {self.config.trainfrac}"
                  "of training data used when training model")

        self.model = model
        self.add_data('model', self.model)


class FlexZBoostEstimator(CatEstimator):
    """FlexZBoost-based CatEstimator
    """
    name = 'FlexZBoostEstimator'
    config_options = CatEstimator.config_options.copy()
    config_options.update(nzbins=SHARED_PARAMS,
                          nondetect_val=SHARED_PARAMS,
                          mag_limits=SHARED_PARAMS,
                          bands=SHARED_PARAMS,
                          err_bands=SHARED_PARAMS,
                          ref_band=SHARED_PARAMS,
                          qp_representation=Param(str, "interp", msg="qp generator to use. [interp|flexzboost]")
                          )

    def __init__(self, args, comm=None):
        """ Constructor:
        Do CatEstimator specific initialization """
        CatEstimator.__init__(self, args, comm=comm)
        if self.config.ref_band not in self.config.bands:
            raise ValueError("ref_band not present in bands list! ")
        self.zgrid = None

    def _process_chunk(self, start, end, data, first):
        print(f"Process {self.rank} estimating PZ PDF for rows {start:,} - {end:,}")

        # replace nondetects
        for bandname, errname in zip(self.config.bands, self.config.err_bands):
            if np.isnan(self.config.nondetect_val):  # pragma: no cover
                detmask = np.isnan(data[bandname])
                if isinstance(data, pd.DataFrame):
                    data.loc[detmask, bandname] = self.config.mag_limits[bandname]
                    data.loc[detmask, errname] = 1.0
                else:
                    detmask = np.isnan(data[bandname])
                    data[bandname][detmask] = self.config.mag_limits[bandname]
                    data[errname][detmask] = 1.0
            else:
                detmask = np.isclose(data[bandname], self.config.nondetect_val, atol=0.01)
                if isinstance(data, pd.DataFrame):  # pragma: no cover
                    data.loc[detmask, bandname] = self.config.mag_limits[bandname]
                    data.loc[detmask, errname] = 1.0
                else:
                    data[bandname][detmask] = self.config.mag_limits[bandname]
                    data[errname][detmask] = 1.0

        color_data = make_color_data(data, self.config.bands, self.config.err_bands,
                                     self.config.ref_band)

        ancil_dictionary = dict()

        calculated_point_estimates = []
        if 'calculated_point_estimates' in self.config:
            calculated_point_estimates = self.config.calculated_point_estimates

        if self.config.qp_representation == 'interp':
            pdfs, z_grid = self.model.predict(color_data, n_grid=self.config.nzbins)
            self.zgrid = np.array(z_grid).flatten()

            if 'mode' in calculated_point_estimates:
                ancil_dictionary.update(mode = np.expand_dims(self.zgrid[np.argmax(pdfs, axis=1)], -1))

            qp_dstn = qp.Ensemble(qp.interp, data=dict(xvals=self.zgrid, yvals=pdfs))

            if 'mean' in calculated_point_estimates:
                ancil_dictionary.update(mean = qp_dstn.mean())

            if 'median' in calculated_point_estimates:
                ancil_dictionary.update(median = qp_dstn.median())

        elif self.config.qp_representation == 'flexzboost':
            basis_coefficients = self.model.predict_coefs(color_data)
            qp_dstn = qp.Ensemble(qp_flexzboost.flexzboost_create_from_basis_coef_object,
                                  data=dict(weights=basis_coefficients.coefs,
                                            basis_coefficients_object=basis_coefficients))

            if 'mode' in calculated_point_estimates:
                # `make_grid` is a helper function from Flexcode that will create a nested
                # array of linearly spaced values. We then flatten that nested array.
                # so the final output will have the form `[0.0, 0.1, ..., 3.0]`.
                self.zgrid = np.array(make_grid(self.config.nzbins, basis_coefficients.z_min, basis_coefficients.z_max)).flatten()
                ancil_dictionary.update(mode = qp_dstn.mode(grid=self.zgrid))

            if 'mean' in calculated_point_estimates:
                ancil_dictionary.update(mean = qp_dstn.mean())

            if 'median' in calculated_point_estimates:
                ancil_dictionary.update(median = qp_dstn.median())

        else:
            raise ValueError(f"Unknown qp_representation in config: {self.config.qp_representation}. Should be one of [interp|flexzboost]")

        if calculated_point_estimates:
            qp_dstn.set_ancil(ancil_dictionary)

        self._do_chunk_output(qp_dstn, start, end, first)
