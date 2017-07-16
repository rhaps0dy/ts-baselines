from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.numpy2ri
import numpy as np
from sklearn.mixtures import BayesianGaussianMixture
from scipy.special import digamma

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


def _estimate_missing_gaussian_parameters(X, resp, reg_covar, covariance_type):
    if covariance_type != "full":
        raise NotImplementedError
    all_missing = np.isnan(X)
    for i in range(len(X)):
        missing = all_missing[i, :]
        y = X[i, :]

    per_feature_resp = resp.expand_dims(2).tile([1, 1, X.shape[1]])
    per_feature_resp[:, missing] = 0
    nk = per_feature_resp.sum(axis=1) + 10 * np.finfo(resp.dtype).eps
    per_feature_resp *= X
    means = per_feature_resp.nansum(axis=1) / nk
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps


def _estimate_missing_log_gaussian_prob(*args):
    raise NotImplemented


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
            self.covariance_prior = estimate_covariance(X)
        super(BayesianMixtureMissingData, self)._check_parameters(X)

    def _initialize(self, X, resp):
        nk, xk, sk = _estimate_missing_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _m_step(self, X, log_resp):
        n_samples, _ = X.shape
        nk, xk, sk = _estimate_missing_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_missing_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (self.degrees_of_freedom_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / self.mean_precision_)
