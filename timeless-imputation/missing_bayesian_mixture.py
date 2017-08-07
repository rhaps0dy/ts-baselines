from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.numpy2ri
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.special import digamma
from gmm_impute import mask_matrix
import gmm_impute
import pandas as pd
import utils
import datasets
import pickle_utils as pu
import os
from tqdm import tqdm


rpy2.robjects.numpy2ri.activate()
py2ri = rpy2.robjects.numpy2ri.py2ri
ri2py = rpy2.robjects.numpy2ri.ri2py

Rstats = importr("stats")


def estimate_covariance(X):
    # Estimate covariance matrix for X with missing data
    # Could use http://journals.sagepub.com/doi/pdf/10.3102/10769986024001021
    # But this is easier
    m = ri2py(Rstats.cov(x=py2ri(X), use="pairwise.complete.obs"))
    # Make symmetric
    m = (m + m.T) / 2
    assert np.all(np.linalg.eigvals(m) >= 0), "m is not positive semi-definite"
    return m


def _estimate_missing_gaussian_parameters(X, resp, means, covariances,
                                          reg_covar, covariance_type):
    if covariance_type != "full":
        raise NotImplementedError
    all_missing = np.isnan(X)
    next_means = np.zeros_like(means)
    next_means_candidate = np.zeros_like(means)
    next_covariances = np.zeros((X.shape[0],) + covariances.shape,
                                dtype=covariances.dtype)

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    nk_vert = nk[:, np.newaxis]
    dependent_covs = [None] * len(X)
    dependent_data_means = np.zeros((X.shape[0],) + means.shape,
                                    dtype=means.dtype)
    for i in range(len(X)):
        missing = all_missing[i, :]
        y = X[i, ~missing]
        K_misobs = mask_matrix(covariances, missing, ~missing)
        K_obsobs = mask_matrix(covariances, ~missing, ~missing)
        K_1222 = K_misobs @ np.linalg.inv(K_obsobs)

        means_vert = means[:, missing, np.newaxis]
        mean_missing = means_vert + K_1222 @ (
            y[:, np.newaxis] - means[:, ~missing, np.newaxis])
        next_means_candidate[:, missing] = mean_missing.squeeze(2)
        next_means_candidate[:, ~missing] = y
        next_means += (resp[i, :, np.newaxis] * next_means_candidate) / nk_vert

        K_mismis = mask_matrix(covariances, missing, missing)
        cov = K_mismis - K_1222 @ K_misobs.transpose([0, 2, 1])
        dependent_covs[i] = cov
        dependent_data_means[i, ...] = next_means_candidate

    for i, cov in enumerate(dependent_covs):
        centered_point = dependent_data_means[i] - next_means
        next_covariances[i, ...] = (np.expand_dims(centered_point, 1) *
                                    np.expand_dims(centered_point, 2))

        missing = all_missing[i, :]
        # `dependent_means_offsetted` ought to be the dependent means
        # (dependent on the observed variables), offsetted by the next means.
        dependent_means_offsetted = centered_point[:, missing]
        M = np.concatenate([np.linalg.cholesky(cov),
                            np.expand_dims(dependent_means_offsetted, 2)],
                           axis=2)
        mismis_covs = M @ M.transpose([0, 2, 1])
        for j, k in enumerate(np.nonzero(missing)[0]):
            for l in range(next_covariances.shape[1]):
                next_covariances[i, l, k, missing] = mismis_covs[l, j, :]

    next_covariances *= resp[:, :, np.newaxis, np.newaxis]
    next_covariances /= nk_vert[:, :, np.newaxis]
    next_covariances = next_covariances.sum(axis=0)
    for k in range(next_covariances.shape[0]):
        next_covariances[k].flat[::next_covariances.shape[1] + 1] += reg_covar
    return nk, next_means, next_covariances


def _estimate_missing_log_gaussian_prob(X, means, covariances):
    # We can take half the log determinant of a covariance by summing
    # over the masked members of this vector (one per component)
    covs_chol_diagonal = np.zeros_like(means)
    for k, m in enumerate(np.linalg.cholesky(covariances)):
        covs_chol_diagonal[k] = m.flat[::m.shape[0]+1]

    out = np.zeros([X.shape[0], covariances.shape[0]], dtype=X.dtype)
    for i, inputs in enumerate(X):
        _mask = ~np.isnan(inputs)
        n_features = np.sum(_mask, axis=0, keepdims=True)

        diffs = inputs[_mask] - means[:, _mask]
        right_diffs = np.expand_dims(diffs, axis=2)
        left_diffs = np.expand_dims(diffs, axis=1)
        _covs = mask_matrix(covariances, _mask, _mask)
        log_prob = left_diffs @ np.linalg.inv(_covs) @ right_diffs
        log_prob = np.squeeze(log_prob, axis=[1, 2])

        log_det = covs_chol_diagonal[:, _mask].sum(axis=1)
        out[i, :] = -.5*(n_features * np.log(2 * np.pi) + log_prob) - log_det
    return out


class BayesianMixtureMissingData(BayesianGaussianMixture):
    def __init__(self, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None,
                 mean_precision_prior=None, mean_prior=None,
                 degrees_of_freedom_prior=None, covariance_prior=None,
                 random_state=None, warm_start=False, verbose=0,
                 verbose_interval=10):
        super(BayesianMixtureMissingData, self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

    def _check_parameters(self, X):
        if self.mean_prior is None:
            self.mean_prior = np.nanmean(X, axis=0)
        if self.covariance_prior is None:
            # self.covariance_prior = estimate_covariance(X)
            self.covariance_prior = np.diag(np.nanvar(X, axis=0))
        self.covariances_ = np.stack([self.covariance_prior] *
                                     self.n_components)
        self.means_ = np.stack([self.mean_prior] * self.n_components)
        super(BayesianMixtureMissingData, self)._check_parameters(X)

    def _initialize(self, X, resp):
        nk, xk, sk = _estimate_missing_gaussian_parameters(
            X, resp, self.means_, self.covariances_,
            self.reg_covar, self.covariance_type)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _m_step(self, X, log_resp):
        self._tqdm.update()
        n_samples, _ = X.shape
        nk, xk, sk = _estimate_missing_gaussian_parameters(
            X, np.exp(log_resp), self.means_, self.covariances_,
            self.reg_covar, self.covariance_type)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_missing_log_gaussian_prob(
            X, self.means_, self.covariances_) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (self.degrees_of_freedom_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / self.mean_precision_)


def impute_bayes_gmm(log_path, dataset, number_imputations=100, full_data=None,
                     n_components=10, n_init=5, init_params='random'):
    del full_data
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    df, cat_idx = dataset
    df = df.copy()
    naive_fillna = {}
    prev_int_keys = []
    for k in df.keys():
        if df[k].dtype == np.int32:
            l = list(df[k].unique())
            if utils.NA_int32 in l:
                l.remove(utils.NA_int32)
            if len(l) > 2:
                naive_fillna[k] = l[0]
            else:
                not_NAs = df[k] != utils.NA_int32
                df[k] = df[k].astype(np.float64).where(not_NAs)
                prev_int_keys.append(k)

    keys = list(filter(lambda k: k not in naive_fillna,
                       df.keys()))
    data = df[keys].values
    params_path = os.path.join(log_path, 'params.pkl.gz')
    if os.path.exists(params_path):
        d = pu.load(params_path)
    else:
        m = BayesianMixtureMissingData(n_components=n_components,
                                       n_init=n_init,
                                       init_params=init_params)
        m._tqdm = tqdm()
        m.fit(data)
        d = {}
        for attr in ['weights', 'means', 'covariances']:
            d[attr] = getattr(m, attr+'_')
        pu.dump(d, params_path)
        del m._tqdm
        pu.dump(m, os.path.join(log_path, 'model.pkl.gz'))
        del m

    _id = gmm_impute._gmm_impute(d, data, n_impute=number_imputations)
    imputed_data = np.mean(_id, axis=0)

    imputed_df = datasets.dataframe_like(df[keys], imputed_data)
    imputed_df[prev_int_keys] = imputed_df[prev_int_keys].apply(
        lambda a: np.round(a).astype(np.int32))
    for k, value in naive_fillna.items():
        df[k] = df[k].where(df[k] != utils.NA_int32, other=value)
    ret = [pd.concat([imputed_df, df[list(naive_fillna.keys())]],
                     axis=1)[df.keys()]]
    impute_path = os.path.join(log_path, "imputed.pkl.gz")
    if not os.path.exists(impute_path):
        pu.dump(ret, impute_path)
    return ret


def mf_initial_impute(log_path, df, info, n_components=15, n_init=5,
                      init_params='random', ignore_categories=False):
    params_path = os.path.join(log_path, 'params.pkl.gz')
    if os.path.exists(params_path):
        d = pu.load(params_path)
    else:
        m = BayesianMixtureMissingData(n_components=n_components,
                                       n_init=n_init,
                                       init_params=init_params)
        m._tqdm = tqdm()
        m.fit(df.values)
        d = {}
        for attr in ['weights', 'means', 'covariances']:
            d[attr] = getattr(m, attr+'_')
        pu.dump(d, params_path)
        del m._tqdm
        pu.dump(m, os.path.join(log_path, 'model.pkl.gz'))
        del m

    # We will estimate the integral of the softmax so we take a sample of the
    # categorical variables:

    if ignore_categories:
        categoric_indices = False
        n_impute = 1
    else:
        categoric_indices = []
        df_keys = list(df.keys())
        cat_keys = list(info['cat_dummies'].keys())
        for k in cat_keys:
            for kk in info['cat_dummies'][k]:
                categoric_indices.append(df_keys.index(kk))
        n_impute = 100000
    _id, var_id = gmm_impute._gmm_impute(d, df.values, n_impute=n_impute,
                                         sample_impute=categoric_indices)
    if not ignore_categories:
        for k in cat_keys:
            # Monte-Carlo integral of softmax
            if len(info['cat_dummies']) > 1:
                min_i = 1000000
                max_i = 0
                for kk in info['cat_dummies'][k]:
                    min_i = min(min_i, df_keys.index(kk))
                    max_i = max(max_i, df_keys.index(kk))
                print(k, min_i, max_i, df_keys)
                _id[:, min_i:max_i+1] = np.exp(_id[:, min_i:max_i+1])
                _id[:, min_i:max_i+1] /= np.sum(_id[:, min_i:max_i+1], axis=2, keepdims=True)
            else:
                i = cat_keys.index(kk)
                _id[:, i] = np.round(_id[:, i]).clip(0, 1)

    imputed_data = np.mean(_id, axis=0)
    imputed_df = datasets.dataframe_like(df, imputed_data)
    variance_df = datasets.dataframe_like(df, var_id)

    return imputed_df, variance_df, d


if __name__ == 'OLD__main__':
    full_data = datasets.datasets()["BostonHousing"][0]
    amputed_data = pu.load(
        "impute_benchmark/amputed_BostonHousing_MCAR_total_0.3.pkl.gz")
    # make ints into float
    NAs = amputed_data.chas != utils.NA_int32
    amputed_data.chas = amputed_data.chas.astype(np.float64).where(NAs)
    full_data.chas = full_data.chas.astype(np.float64)
    (_ad, *_), moments = utils.normalise_dataframes(amputed_data)

    m = BayesianMixtureMissingData(n_components=10, n_init=5,
                                   init_params='random')
    m.fit(_ad.values)
    d = {}
    for attr in ['weights', 'means', 'covariances']:
        d[attr] = getattr(m, attr+'_')
    _id = gmm_impute._gmm_impute(d, _ad.values)
    imputed_data = utils.unnormalise_dataframes(moments, _id)

    rmse_fd, rmse_ad, *rmse_id = utils.normalise_dataframes(
        full_data, amputed_data, *imputed_data, method='min_max')[0]

    print("RMSE:", utils.mean_rmse(
        np.isnan(rmse_ad.values), rmse_fd.values,
        list(d.values for d in rmse_id)))

if __name__ == '__main__':
    _ds = datasets.datasets()
    dsets = dict(filter(lambda t: t[0] in {"Ionosphere"},
                        _ds.items()))
    baseline = datasets.benchmark({
        'BGMM_20': lambda p, d, full_data: impute_bayes_gmm(
            p, d, full_data=full_data, number_imputations=100,
            n_components=20),
        'MissForest': datasets.memoize(utils.impute_missforest),
    }, dsets, do_not_compute=False)
    print(baseline)
