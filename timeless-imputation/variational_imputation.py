import autograd.numpy as np
import autograd.scipy.special as sp
from autograd import grad
from autograd.optimizers import adam
import os
from autograd.core import primitive
from autograd.scipy.misc import logsumexp


def kl_Z_V(gamma, logit_resp, alpha):
    a = sp.digamma(gamma[:, 0])
    b = sp.digamma(gamma[:, 1])
    c_arg = np.sum(gamma, axis=1)
    c = sp.digamma(c_arg)
    # Keep resp in [0, 1] and summing to 1
    log_resp = logit_resp - logsumexp(logit_resp, axis=1)[:, np.newaxis]
    resp = np.exp(log_resp)
    resp_k = np.sum(resp[:, :-1], axis=0)
    resp_cum = np.sum(np.cumsum(resp[:, ::-1], axis=1)[:, -2::-1], axis=0)
    e_log_p = np.sum(resp_cum*(a - c) + resp_k*(b - c))
    e_log_q = np.sum(resp * log_resp)

    kl_V = np.sum((gamma[:, 0] - 1)*a
                  + (gamma[:, 1] - alpha)*b
                  + (1 - gamma[:, 0] + alpha - gamma[:, 1])*c
                  - sp.gammaln(gamma[:, 0]) - sp.gammaln(gamma[:, 1])
                  + sp.gammaln(c_arg)
                  #+ sp.gammaln(1) + sp.gammaln(alpha) - sp.gammaln(1+alpha)
                  , axis=0)
    return e_log_q - e_log_p + kl_V


def kl_NW(means, sqrt_beta, precs, sqrt_remain_deg_free, p_means, p_beta,
          p_covs, p_deg_free):
    D = means.shape[1]

    # Prevent beta from being negative
    beta = sqrt_beta**2
    # ensure nu > D-1
    deg_free = sqrt_remain_deg_free**2 + (D - 0.99)

    mean_dist = (p_means - means)[:, :, np.newaxis]
    mahalanobis_term = p_beta/2 * np.sum(np.matmul(np.matmul(
        np.transpose(mean_dist, (0, 2, 1)), precs), mean_dist))
    beta_q = p_beta / beta
    beta_term = D/2 * np.sum(beta_q - np.log(beta_q))

    wishart_prec_term = p_deg_free/2*(-np.sum(np.log(np.linalg.det(precs)))
                                      + np.sum(p_covs * precs))
    deg_free_term = np.sum(p_deg_free)*D / (-2.)

    di_gamma_arg = (deg_free[:, np.newaxis] - np.arange(D))/2
    digamma_term = np.sum((deg_free - p_deg_free)/2 *
                          np.sum(sp.digamma(di_gamma_arg), axis=1), axis=0)
    gamma_term = -np.sum(sp.gamma(di_gamma_arg))
    return (mahalanobis_term + beta_term + wishart_prec_term + deg_free_term +
            gamma_term + digamma_term)


def kl_div(params, prior, X):
    _kl_Z_V = kl_Z_V(params['gamma'], params['logit_resp'], prior['alpha'])
    _kl_NW = kl_NW(params['means'], params['beta'], params['precs'],
                   params['deg_free'], prior['means'], prior['beta'],
                   prior['covs'], prior['deg_free'])
    _kl_likelihood = 0.
    return _kl_Z_V + _kl_NW + _kl_likelihood


class VariationalImputation:
    def __init__(self, n_components, n_init, alpha_prior=None):
        self.n_components = n_components
        self.n_init = n_init
        if alpha_prior is None:
            alpha_prior = 1/n_components
        self.alpha_prior = alpha_prior

    def make_prior(self, X):
        """This is a Dirichlet Process prior with a Gaussian-Wishart
        distribution."""
        _, D = X.shape
        del X
        prior = dict(
            alpha=self.alpha_prior,
            means=np.zeros(shape=(D,), dtype=np.float64),
            beta=1.,
            precs=np.eye(D, dtype=np.float64),
            deg_free=D,
        )
        prior['covs'] = np.linalg.inv(prior['precs'])
        return prior

    def make_params(self, X, prior):
        N, D = X.shape
        K = self.n_components
        params = dict(
            mis_means=np.zeros(shape=(N, D), dtype=np.float64),
            mis_covs=np.zeros(shape=(N, D, D), dtype=np.float64),
            logit_resp=np.zeros(shape=(N, K), dtype=np.float64),
            gamma=np.zeros(shape=(K-1, 2), dtype=np.float64) + 0.5,
            means=np.stack([prior['means']]*K, axis=0),
            beta=np.zeros(shape=(K,), dtype=np.float64) + prior['beta'],
            precs=np.stack([prior['precs']]*K, axis=0),
            deg_free=np.zeros(shape=(K,), dtype=np.float64),
        )
        params['mis_covs'][...] = np.eye(D)
        return params

    def fit(self, X):
        self.prior = self.make_prior(X)
        init_params = self.make_params(X, self.prior)

        def _kl_div(params, _iter):
            return kl_div(params, self.prior, X)

        def callback(params, _iter, gradient):
            if _iter % 100 == 0:
                print("Iteration", _iter,
                      "KL divergence:", _kl_div(params, 0))

        kl_div_grad = grad(_kl_div)
        print("Training...")
        self.trained_params = adam(kl_div_grad, init_params, step_size=0.1,
                                   num_iters=1000, callback=callback)


def initial_impute(log_path, df, full_data, info, n_components=15, n_init=5,
                   ignore_categories=False):
    params_path = os.path.join(log_path, 'params.pkl.gz')
    if False and os.path.exists(params_path):
        d = pu.load(params_path)
    else:
        m = VariationalImputation(n_components, n_init)
        _full_data = full_data.astype(np.float64)
        _full_data.chas -= 1.5
        m.fit(_full_data.values)
