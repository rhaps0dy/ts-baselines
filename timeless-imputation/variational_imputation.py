import autograd.numpy as np
import autograd.scipy.special as sp
from autograd import grad, value_and_grad
from autograd.optimizers import adam
import os
from autograd.core import primitive
from autograd.scipy.misc import logsumexp
from autograd.util import flatten_func
from scipy.optimize import minimize
#import numpy as np_orig


def kl_Z_V(gamma, logit_resp, alpha):
    a = sp.digamma(gamma[:, 0])
    b = sp.digamma(gamma[:, 1])
    c_arg = np.sum(gamma, axis=1)
    c = sp.digamma(c_arg)
    # Keep resp in [0, 1] and summing to 1
    log_resp = logit_resp - logsumexp(logit_resp, axis=1)[:, np.newaxis]
    resp = np.exp(log_resp)
    resp_k = np.sum(resp, axis=0)
    resp_cum = np.sum(np.cumsum(resp[:, ::-1], axis=1)[:, -2::-1], axis=0)
    e_log_p = np.sum(resp_cum*(a - c) + resp_k[:-1]*(b - c))
    e_log_q = np.sum(resp * log_resp)

    kl_V = np.sum((gamma[:, 0] - 1)*a
                  + (gamma[:, 1] - alpha)*b
                  + (1 - gamma[:, 0] + alpha - gamma[:, 1])*c
                  - sp.gammaln(gamma[:, 0]) - sp.gammaln(gamma[:, 1])
                  + sp.gammaln(c_arg)
                  #+ sp.gammaln(1) + sp.gammaln(alpha) - sp.gammaln(1+alpha)
                  , axis=0)
    return e_log_q - e_log_p + kl_V, resp, resp_k


def kl_NW(means, beta, precs, deg_free, p_means, p_beta, p_covs, p_deg_free, N,
          resp_k):
    D = means.shape[1]

    mean_dist = (p_means - means)[:, :, np.newaxis]
    mahalanobis_term = p_beta/2 * np.sum(np.matmul(np.matmul(
        np.transpose(mean_dist, (0, 2, 1)), precs), mean_dist))
    beta_q = p_beta / beta
    beta_term = D/2 * np.sum(beta_q - np.log(beta_q))

    #det = np.clip(np.linalg.det(precs), np.finfo(np.float64).tiny, np.inf)
    log_det = np.log(np.linalg.det(precs))
    wishart_prec_term = p_deg_free/2*(-np.sum(log_det)
                                      + np.sum(p_covs * precs))
    likelihood_log_det_term = np.sum(resp_k * (log_det + D/beta))/2
    deg_free_term = np.sum(p_deg_free)*D / (-2.)

    di_gamma_arg = (deg_free[:, np.newaxis] - np.arange(D))/2
    # We add +resp_k because of the term in KL-div of normals
    digamma_term = np.sum((deg_free - p_deg_free - resp_k)/2 *
                          np.sum(sp.digamma(di_gamma_arg), axis=1), axis=0)
    gamma_term = -np.sum(sp.gamma(di_gamma_arg))
    return (mahalanobis_term + beta_term + wishart_prec_term + deg_free_term +
            gamma_term + digamma_term + likelihood_log_det_term)


#np.seterr(all='raise')
def kl_likelihood(mis_means, mis_covs, resp, resp_k, X, means, beta, precs,
                  deg_free, mask):
    N, D = X.shape
    K, _ = means.shape
    t = 0.
    resp_n = np.sum(resp, axis=1)
    for i in range(N):
        mis = mask[i, :]
        n_mis = np.sum(mis)
        mis_cov_i_H = (mis_covs[i, :n_mis, :n_mis]
                       + np.eye(n_mis)*np.finfo(np.float64).eps)
        mis_cov_i = np.matmul(np.transpose(mis_cov_i_H), mis_cov_i_H)
        #det = np.clip(np.linalg.det(mis_cov_i), np.finfo(np.float64).tiny,
        #              np.inf)
        t = t - resp_n[i] * np.log(np.linalg.det(mis_cov_i))

        diff = (mis_means[i, :]*mis + X[i, :]) - means
        mahalanobis = np.matmul(np.matmul(diff[:, np.newaxis, :], precs),
                                          diff[:, :, np.newaxis])
        vec = ((deg_free - D + n_mis) * np.sum(precs[:, mis, :][:, :, mis]
                                               * mis_cov_i, axis=(1, 2))
               + deg_free * np.squeeze(mahalanobis, axis=(1, 2)))
        t = t + np.dot(resp[i, :], vec) # + resp_n[i]*n_mis
    #t = t + N*D*np.log(2)
    return .5*t


def kl_div(params, prior, X, mask):
    _kl_Z_V, resp, resp_k = kl_Z_V(params['gamma'], params['logit_resp'],
                                   prior['alpha'])
    N, D = X.shape

    EPS = np.finfo(np.float64).eps

    # Prevent beta from being negative
    beta = params['beta']**2
    #beta = np.zeros_like(params['beta']) + prior['beta']
    # ensure nu > D-1
    deg_free = params['deg_free']**2 + D*(1. - EPS)
    #deg_free = np.zeros_like(params['deg_free']) + prior['deg_free']

    precs = np.matmul(np.transpose(params['precs'], (0, 2, 1)),
                      params['precs']) + np.eye(D)*EPS
    #print("min precs", np.abs(precs).min(), "min miscovs", np.abs(params['mis_covs']).min())
    if not np.all(np.isfinite(precs)):
        import pdb
        pdb.set_trace()
    #precs = np.zeros((15, D, D)) #_like(params['precs'])
    #precs[...] = np.eye(D)
    #mis_covs = np.zeros((N, D, D)) #_like(params['precs'])
    #mis_covs[...] = np.eye(D)

    _kl_NW = kl_NW(params['means'], beta, precs,
                   deg_free, prior['means'], prior['beta'],
                   prior['covs'], prior['deg_free'], len(X), resp_k)

    _kl_likelihood = kl_likelihood(params['mis_means'], params['mis_covs'],
                                   resp, resp_k, X, params['means'], beta,
                                   precs, deg_free, mask)
    return _kl_NW + _kl_likelihood + _kl_Z_V #+ _kl_NW # + _kl_likelihood


class VariationalImputation:
    def __init__(self, n_components, n_init, alpha_prior=None):
        self.n_components = n_components
        self.n_init = n_init
        if alpha_prior is None:
            alpha_prior = 1/n_components
        self.alpha_prior = alpha_prior

    def make_prior(self, mask):
        """This is a Dirichlet Process prior with a Gaussian-Wishart
        distribution."""
        N, D = mask.shape
        K = self.n_components
        prior = dict(
            alpha=self.alpha_prior,
            means=np.zeros(shape=(D,), dtype=np.float64),
            beta=1.,
            precs=np.eye(D, dtype=np.float64),
            #logit_resp=np.zeros(shape=(N, K), dtype=np.float64),
            deg_free=D,
        )
        prior['covs'] = np.linalg.inv(prior['precs'])
        return prior

    def make_params(self, mask, prior):
        N, D = mask.shape
        K = self.n_components
        params = dict(
            mis_means=np.zeros(shape=(N, D), dtype=np.float64),
            mis_covs=np.random.rand(N, D, D).astype(np.float64),
            logit_resp=np.zeros(shape=(N, K), dtype=np.float64),
            gamma=np.zeros(shape=(K-1, 2), dtype=np.float64) + 0.5,
            means=np.stack([prior['means']]*K, axis=0),
            beta=np.zeros(shape=(K,), dtype=np.float64) + prior['beta'],
            precs=np.random.rand(K, D, D).astype(np.float64),
            deg_free=np.zeros(shape=(K,), dtype=np.float64),
        )
        return params

    def fit(self, X):
        mask = np.isnan(X)
        X = X.copy()
        X[mask] = 0.
        assert not np.any(np.isnan(X))

        for init_i in range(self.n_init):
            self.prior = self.make_prior(mask)
            init_params = self.make_params(mask, self.prior)

            def _kl_div(params):
                return kl_div(params, self.prior, X, mask)

            #def callback(params, _iter, gradient):
            #    if _iter % 10 == 0:
            #        print("Iteration", _iter,
            #              "KL divergence:", _kl_div(params, 0))

            flattened_kld, unflatten, x0 = flatten_func(_kl_div, init_params)
            _kl_div_vg = value_and_grad(flattened_kld)
            def kl_div_vg(flat_params):
                v, g = _kl_div_vg(flat_params)
                #g_norm = np.linalg.norm(g, ord=np.inf)
                #print("function value:", v, "gradient norm:", g_norm)
                if np.any(np.isnan(g)): # or g_norm >= 1e7:
                    print(unflatten(g))
                    import pdb
                    pdb.set_trace()
                return v, g #np.clip(g, -1e5, 1e5)

            print("Training...")
            r = minimize(kl_div_vg, x0=x0, jac=True, method='L-BFGS-B',
                         options=dict(maxiter=1000, disp=True))
            import pdb
            pdb.set_trace()
            self.trained_params = unflatten(r.x)

            #kl_div_grad = grad(_kl_div)
            #self.trained_params = adam(kl_div_grad, init_params, step_size=0.1,
            #                           num_iters=1000, callback=callback)


def initial_impute(log_path, df, full_data, info, n_components=15, n_init=5,
                   ignore_categories=False):
    params_path = os.path.join(log_path, 'params.pkl.gz')
    if False and os.path.exists(params_path):
        d = pu.load(params_path)
    else:
        m = VariationalImputation(n_components, n_init)
        #_full_data = full_data.astype(np.float64)
        #_full_data.chas -= 1.5
        df.chas -= 0.5
        m.fit(df.values)
