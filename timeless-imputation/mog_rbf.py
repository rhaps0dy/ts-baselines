import numpy as np
import gmm_impute as gmm
import unittest
from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
from paramz.caching import Cache_this

def tp(x):
    return np.swapaxes(x, -1, -2)

def uncertain_point(inp, mog, cutoff, M, expand_dims=False,
                    single_gaussian_moment_matching=False):
    mis = np.isnan(inp)
    d = gmm.conditional_mog(mog, inp, mis, cutoff=cutoff)
    if single_gaussian_moment_matching:
    # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        ms = d['means'] * d['weights'][:, np.newaxis]
        mean = np.sum(ms, axis=0)
        var_mix = np.sum(d['covariances'] * np.eye(d['covariances'].shape[1]), axis=2).sum(axis=0)
        var = var_mix + np.sum(ms * d['means'], axis=0) - mean
        d['weights'] = np.ones([1], dtype=d['weights'].dtype)
        d['means'] = mean[np.newaxis, :]
        d['covariances'] = np.diag(var_mix)[np.newaxis, :, :]

    d['mis'] = mis
    d['mu_obs'] = inp[~mis][np.newaxis, :, np.newaxis]
    p = np.linalg.inv(d['covariances'])
    d['precisions'] = p

    d['means'] = d['means'][:, :, np.newaxis]
    d['sig_mu'] = p @ d['means']
    d['mu_sig_mu'] = tp(d['means']) @ d['sig_mu']

    A = d['precisions'] + M[d['mis'], :][:, d['mis']]
    d['Ainv'] = np.linalg.inv(A)
    d['sqrt_det'] = (np.linalg.det(d['covariances'])
                    * np.linalg.det(A))**(-.5)
    d['neg_Ainv_a'] = d['Ainv'] @ d['sig_mu']
    d['a_Ainv_a'] = tp(d['sig_mu']) @ d['neg_Ainv_a']
    return d


def rbf_uncertain(x, v, M_):
    M = M_[np.newaxis, np.newaxis, :, :]
    misx = x['mis']
    misv = v['mis']
    obsx = ~misx
    obsv = ~misv

    x_mu_obs = x['mu_obs'][:, np.newaxis]
    v_mu_obs = v['mu_obs'][np.newaxis, :]
    Ainv = x['Ainv'][:, np.newaxis]
    Cinv = v['Ainv'][np.newaxis, :]
    a_Ainv_a = x['a_Ainv_a'][:, np.newaxis]
    neg_Ainv_a = x['neg_Ainv_a'][:, np.newaxis]
    x_sig_mu = x['sig_mu'][:, np.newaxis]
    v_sig_mu = v['sig_mu'][np.newaxis, :]
    x_mu_sig_mu = x['mu_sig_mu'][:, np.newaxis]
    v_mu_sig_mu = v['mu_sig_mu'][np.newaxis, :]

    ww = (x['weights'][:, np.newaxis]
          * v['weights'][np.newaxis, :]
          * x['sqrt_det'][:, np.newaxis]
          * v['sqrt_det'][np.newaxis, :])

    exp = x_mu_sig_mu + v_mu_sig_mu - a_Ainv_a
    exp += tp(x_mu_obs) @ M[:, :, obsx, :][:, :, :, obsx] @ x_mu_obs

    b = M[:, :, :, misx] @ neg_Ainv_a + M[:, :, :, obsx] @ x_mu_obs
    B_2 = -(M - M[:, :, :, misx] @ Ainv @ M[:, :, misx, :])
    d = B_2[:, :, :, obsv] @ v_mu_obs
    c = v_sig_mu + b[:, :, misv] + d[:, :, misv]

    exp -= tp(c) @ Cinv @ c
    exp -= tp(2*b + d)[:, :, :, obsv] @ v_mu_obs

    exp = np.squeeze(exp, (-2, -1))
    assert exp.shape == ww.shape

    return np.sum(ww * np.exp(-.5*exp))


class UncertainMogRBFWhite(Kern):
    def __init__(self, input_dim, mog, white_var=1., rbf_var=1.,
                 lengthscale=1., ARD=False, cutoff=0.99, single_gaussian=False,
                 active_dims=None, name='uncertainMoG'):
        """For this kernel, to save computation, we are going to assume the
        dimension 0 of the inputs bears an ID of the point. Thus, X.shape[1] ==
        input_dim + 1.
        `cutoff` represents the % of the variance that the MoG must represent
        before it is cut off."""
        super(UncertainMogRBFWhite, self).__init__(input_dim, active_dims, name)
        self.input_dim = input_dim
        self.mog = mog
        self.single_gaussian = single_gaussian
        assert self.mog['means'].shape[1] == input_dim
        self.cutoff = cutoff

        self.white_var = Param('white_var', white_var)
        self.rbf_var = Param('rbf_var', rbf_var)
        self.lengthscale = Param('lengthscale', lengthscale)
        assert len(self.lengthscale.shape) == 1, \
                "Lengthscale must be of rank 1"
        assert (self.lengthscale.shape[0] == 1 or
                self.lengthscale.shape[0] == self.input_dim), \
                "The lengthscale must be a single number or a diagonal"
        self.link_parameters(self.white_var, self.rbf_var, self.lengthscale)

        self.M = np.zeros([input_dim, input_dim])
        self.parameters_changed()

    def parameters_changed(self):
        self.M.flat[::self.M.shape[-1]+1] = self.lengthscale

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        X = list(map(lambda inp: uncertain_point(
            inp, self.mog, self.cutoff, self.M,
            single_gaussian_moment_matching=self.single_gaussian), X))

        if X2 is None:
            out = np.zeros([len(X), len(X)])
            for i, x in enumerate(X):
                for j in range(i+1, len(X)):
                    out[i, j] = rbf_uncertain(x, X[j], self.M)
            out += out.T
            out *= self.rbf_var
            out.flat[::len(X)+1] = self.rbf_var + self.white_var
        else:
            X2 = list(map(lambda inp: uncertain_point(
                inp, self.mog, self.cutoff, self.M,
                single_gaussian_moment_matching=self.single_gaussian), X2))
            out = np.zeros([len(X), len(X2)])
            for i, x in enumerate(X):
                for j, x2 in enumerate(X2):
                    out[i, j] = rbf_uncertain(x, x2, self.M)
            out *= self.rbf_var
        return out

    def Kdiag(self, X):
        return (self.rbf_var + self.white_var) * np.ones(len(X))

    def update_gradients_full(self, dL_dK, X, X2):
        pass


if __name__ == '__main__':
    import pickle_utils as pu

    @pu.memoize('DATA_GOTTEN.pkl.gz')
    def get_data():
        import missForest
        import datasets
        import category_dae
        bh, cats = datasets.datasets()["BostonHousing"]
        info = category_dae.dataset_dimensions_info((bh, cats))
        test_df, masks_usable = missForest.preprocess_dataframe(
            bh, info, ignore_ordered=True)
        return test_df

    #test_df = get_data()
    test_df = pu.load("amputed_BostonHousing_MCAR_total_0.3.pkl.gz")
    test_df = (test_df - test_df.mean()) / test_df.std()
    m = pu.load(#"impute_benchmark/imputed_GMM_BostonHousing_MCAR_total_0.3/"
                "params.pkl.gz")
    assert len(test_df.values[0].shape) == 1

    M = np.ones(test_df.shape[1]) / 1000
    x = uncertain_point(m, test_df.values[0], M)
    x['id'] = 0
    v = uncertain_point(m, test_df.values[1], M)
    v['id'] = 1

    print(rbf_uncertain(x, v, M))
    print(rbf_uncertain(v, x, M))
