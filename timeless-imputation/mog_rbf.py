import autograd.numpy as np
import autograd

import gmm_impute as gmm
import unittest
from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
from paramz.caching import Cache_this
import time

def tp(x):
    return np.swapaxes(x, -1, -2)

def uncertain_point(inp, mog, cutoff, M, expand_dims=False,
                    single_gaussian_moment_matching=False):
    mis = np.isnan(inp)
    d = gmm.conditional_mog(mog, inp, mis, cutoff=cutoff)
    return with_d_uncertain_point(d, inp, mis, M, expand_dims,
                            single_gaussian_moment_matching)


def with_d_uncertain_point(d, inp, mis, M, expand_dims=False,
                     single_gaussian_moment_matching=False):
    if single_gaussian_moment_matching:
        mean, var = gmm.single_gaussian_moment_matching(d)
        d['means'] = mean[np.newaxis, :]
        d['covariances'] = np.diag(var)[np.newaxis, :, :]
        d['weights'] = np.ones([1], dtype=d['weights'].dtype)

    d['mis'] = mis
    obs = ~mis
    d['mu_obs'] = inp[obs][np.newaxis, :, np.newaxis]
    p = np.linalg.inv(d['covariances'])
    d['precisions'] = p

    d['means'] = d['means'][:, :, np.newaxis]
    d['sig_mu'] = p @ d['means']
    d['mu_sig_mu'] = tp(d['means']) @ d['sig_mu']

    A = d['precisions'] + M[d['mis'], :][:, d['mis']]
    d['Ainv'] = np.linalg.inv(A)
    sqrt_det = (np.linalg.det(d['covariances'])
                * np.linalg.det(A))**(-.5)
    d['norm'] = d['weights'] * sqrt_det
    neg_Ainv_a = d['Ainv'] @ d['sig_mu']
    a_Ainv_a = tp(d['sig_mu']) @ neg_Ainv_a

    obs_sc = tp(d['mu_obs']) @ M[obs, :][:, obs] @ d['mu_obs']
    d['sum_contrib'] = d['mu_sig_mu'] - a_Ainv_a + obs_sc
    d['b'] = M[:, mis] @ neg_Ainv_a + M[:, obs] @ d['mu_obs']
    d['B_2'] = -(M - M[:, mis] @ d['Ainv'] @ M[mis, :])
    return d


def rbf_uncertain(x, v, M_):
    M = M_[np.newaxis, np.newaxis, :, :]
    misv = v['mis']
    obsv = ~misv

    v_mu_obs = v['mu_obs'][np.newaxis, :]
    v_sig_mu = v['sig_mu'][np.newaxis, :]
    Cinv = v['Ainv'][np.newaxis, :]

    b = x['b'][:, np.newaxis]
    B_2 = x['B_2'][:, np.newaxis]
    x_sum_contrib = x['sum_contrib'][:, np.newaxis]
    v_mu_sig_mu = v['mu_sig_mu'][np.newaxis, :]

    ww = (x['norm'][:, np.newaxis]
          * v['norm'][np.newaxis, :])

    d = B_2[:, :, :, obsv] @ v_mu_obs
    c = v_sig_mu + b[:, :, misv] + d[:, :, misv]
    a = tp(c) @ Cinv @ c
    e = tp(2*b + d)[:, :, :, obsv] @ v_mu_obs
    exp = x_sum_contrib + v_mu_sig_mu - a - e

    exp = np.squeeze(exp, (-2, -1))
    assert exp.shape == ww.shape
    return np.sum(ww * np.exp(-.5*exp))

class UncertainMoGRBFWhite(Kern):
    def __init__(self, input_dim, mog, white_var=1., rbf_var=1.,
                 ARD=False, cutoff=0.99, single_gaussian=False,
                 active_dims=None, name='uncertainMoG'):
        """For this kernel, to save computation, we are going to assume the
        dimension 0 of the inputs bears an ID of the point. Thus, X.shape[1] ==
        input_dim + 1.
        `cutoff` represents the % of the weight that the MoG must represent
        before it is cut off.
        """
        super(UncertainMoGRBFWhite, self).__init__(input_dim, active_dims, name)
        self.input_dim = input_dim
        self.mog = mog
        self.single_gaussian = single_gaussian
        assert self.mog['means'].shape[1] == input_dim
        self.cutoff = cutoff

        self.white_var = Param('white_var', white_var)
        self.rbf_var = Param('rbf_var', rbf_var)
        if ARD:
            lengthscale = np.ones([input_dim], dtype=np.float64)
        else:
            lengthscale = 1.
        self.lengthscale = Param('lengthscale', lengthscale)
        assert len(self.lengthscale.shape) == 1, \
                "Lengthscale must be of rank 1"
        assert (self.lengthscale.shape[0] == 1 or
                self.lengthscale.shape[0] == self.input_dim), \
                "The lengthscale must be a single number or a diagonal"
        self.link_parameters(self.white_var, self.rbf_var, self.lengthscale)

        self._lengthscale_gradient = autograd.grad(self.kernel_for_derivating)

    @Cache_this(limit=3, ignore_args=())
    def lengthscale_gradient(self, *args):
        return self._lengthscale_gradient(*args)

    def kernel_for_derivating(self, lengthscale, dL_dK, X, X2):
        """ Suitable for calling `autograd.grad`, to get the gradient with
        respect to lengthscale """
        if len(lengthscale) == 1:
            M = lengthscale[0]**(-2) * np.eye(self.input_dim)
        else:
            M = np.diag(lengthscale**(-2))
        #start = time.time()
        X = list(map(lambda inp: uncertain_point(
            inp, self.mog, self.cutoff, M,
            single_gaussian_moment_matching=self.single_gaussian), X))
        #print("To compute map for gradient took me:", time.time() - start)

        #start = time.time()
        if X2 is None:
            out = sum(rbf_uncertain(X[i], X[j], M) * (dL_dK[i, j] + dL_dK[j, i])
                      for i in range(len(X))
                      for j in range(i+1, len(X)))
            ret = (self.rbf_var * out
                   + (self.rbf_var + self.white_var) * np.trace(dL_dK))
        else:
            X2 = list(map(lambda inp: uncertain_point(
                inp, self.mog, self.cutoff, M,
                single_gaussian_moment_matching=self.single_gaussian), X2))
            out = sum(rbf_uncertain(X[i], X2[j], M) * dL_dK[i, j]
                      for i in range(len(X))
                      for j in range(len(X2)))
            ret = self.rbf_var * out
        #print("To compute matrix for gradient took me:", time.time() - start)
        return ret


    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        if len(self.lengthscale) == 1:
            M = self.lengthscale[0]**(-2) * np.eye(self.input_dim)
        else:
            M = np.diag(self.lengthscale**(-2))
        #start = time.time()
        X = list(map(lambda inp: uncertain_point(
            inp, self.mog, self.cutoff, M,
            single_gaussian_moment_matching=self.single_gaussian), X))
        #print("To compute map took me:", time.time() - start)
        #import collections
        #counter = collections.Counter(list(map(lambda d: len(d['weights']), X)))
        #print(counter)

        #start = time.time()
        if X2 is None:
            out = np.zeros([len(X), len(X)])
            for i, x in enumerate(X):
                for j in range(i+1, len(X)):
                    out[i, j] = rbf_uncertain(x, X[j], M)
            out += out.T
            out *= self.rbf_var
            #out = (out + out.T) / 2
            out.flat[::len(X)+1] = self.rbf_var + self.white_var
        else:
            X2 = list(map(lambda inp: uncertain_point(
                inp, self.mog, self.cutoff, M,
                single_gaussian_moment_matching=self.single_gaussian), X2))
            out = np.zeros([len(X), len(X2)])
            for i, x in enumerate(X):
                for j, x2 in enumerate(X2):
                    out[i, j] = rbf_uncertain(x, x2, M)
            out *= self.rbf_var
        #print("To compute matrix took me:", time.time() - start)
        return out

    def Kdiag(self, X):
        return (self.rbf_var + self.white_var) * np.ones(len(X))

    def update_gradients_full(self, dL_dK, X, X2):
        ker = self.K(X, X2)
        if X2 is None:
            self.white_var.gradient = np.trace(dL_dK)
            ker.flat[::len(X)+1] -= self.white_var
        else:
            self.white_var.gradient = 0.0
        self.rbf_var.gradient = np.sum(dL_dK * ker) / self.rbf_var
        self.lengthscale.gradient = \
            self.lengthscale_gradient(np.array(self.lengthscale), dL_dK, X, X2)


    def update_gradients_diag(self, dL_dKdiag, X):
        self.lengthscale.gradient[...] = 0.0
        self.white_var.gradient = \
            self.rbf_var.gradient = np.sum(dL_dKdiag)


class UncertainGaussianRBFWhite(UncertainMoGRBFWhite):
    def __init__(self, input_dim, mog, white_var=1., rbf_var=1.,
                 lengthscale=1., ARD=False, active_dims=None,
                 name='uncertainGaussian'):
        """
        Same as UncertainMoGRBFWhite but with a single Gaussian.
        """
        self.input_dim = input_dim
        self.mog = mog
        if self.mog is None:
            # The input will be the means and the independent covariances
            self.input_dim *= 2
            assert active_dims is None
        else:
            print("WARNING: you passed a MoG to a GaussianRBFWhite kernel")
        Kern.__init__(self, self.input_dim, active_dims, name)

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

    @staticmethod
    def gather_point_stacks(points):
        n_dims = len(points[0]['mis'])
        n_mixtures = len(points[0]['means'])
        assert n_mixtures == 1, "No point doing this with more mixtures"
        n_points = len(points)

        b = np.concatenate(list(map(lambda p: p['b'], points)), axis=0)
        assert b.shape == (n_points, n_dims, 1)

        B_2 = np.concatenate(list(map(lambda p: p['B_2'], points)), axis=0)
        assert B_2.shape == (n_points, n_dims, n_dims)

        norm = np.concatenate(list(map(lambda p: p['norm'], points)), axis=0)
        assert norm.shape == (n_points,)

        sum_contrib = np.concatenate(list(map(lambda p: p['sum_contrib'], points)),
                                     axis=0)
        assert sum_contrib.shape == (n_points, 1, 1)

        mu_sig_mu = np.concatenate(list(map(lambda p: p['mu_sig_mu'], points)),
                                   axis=0)
        assert mu_sig_mu.shape == (n_points, 1, 1)

        sig_mu = np.zeros_like(b)
        mu_obs = np.zeros_like(b)
        Cinv = np.zeros_like(B_2)

        for i, p in enumerate(points):
            mis = p['mis']
            sig_mu[i, mis, :] = p['sig_mu']
            mu_obs[i, ~mis, :] = p['mu_obs']
            Cinv[i, mis[:, np.newaxis]*mis] = p['Ainv'].reshape(n_mixtures, -1)

        return b, B_2, norm, sum_contrib, mu_sig_mu, sig_mu, mu_obs, Cinv

    def points_statistics(self, X):
        if self.mog is None:
            X_var = X[:, X.shape[1]//2:]
            X = X[:, :X_var.shape[1]]
            assert X.shape == X_var.shape
            mis = X_var != 0.

            X_points = []
            weights = np.ones([1], dtype=X.dtype)
            for i in range(len(X)):
                d = {'weights': weights,
                     'means': X[i:i+1, mis[i]],
                     'covariances': np.diag(X_var[i, mis[i]])[np.newaxis, :, :]}
                X_points.append(with_d_uncertain_point(d, X[i], mis[i], self.M))
        else:
            X_points = list(map(lambda inp: uncertain_point(inp, self.mog,
                1., self.M, single_gaussian_moment_matching=True), X))
        return self.gather_point_stacks(X_points)

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        #start = time.time()
        b, B_2, norm, sum_contrib, mu_sig_mu, sig_mu, mu_obs, Cinv = \
            self.points_statistics(X)

        b = b[:, np.newaxis]
        B_2 = B_2[:, np.newaxis]
        x_norm = norm[:, np.newaxis]
        x_sum_contrib = sum_contrib[:, np.newaxis]

        if X2 is None:
            #print("To compute map took me:", time.time() - start)
            #start = time.time()
            v_sig_mu = sig_mu[np.newaxis, :]
            v_mu_sig_mu = mu_sig_mu[np.newaxis, :]
            v_mu_obs = mu_obs[np.newaxis, :]
            Cinv = Cinv[np.newaxis, :]
            v_norm = norm[np.newaxis, :]
        else:
            del norm, sum_contrib, mu_sig_mu, sig_mu, mu_obs, Cinv
            _, _, v_norm, _, v_mu_sig_mu, v_sig_mu, v_mu_obs, Cinv = \
                self.points_statistics(X2)
            #print("To compute map (rectangular) took me:", time.time() - start)
            #start = time.time()
            v_sig_mu = v_sig_mu[np.newaxis, :]
            v_mu_sig_mu = v_mu_sig_mu[np.newaxis, :]
            v_mu_obs = v_mu_obs[np.newaxis, :]
            Cinv = Cinv[np.newaxis, :]
            v_norm = v_norm[np.newaxis, :]

        d = B_2 @ v_mu_obs
        sum_bd = b + d
        c = v_sig_mu + sum_bd
        a = tp(c) @ Cinv @ c
        e = tp(sum_bd + b) @ v_mu_obs
        exp = x_sum_contrib + v_mu_sig_mu - a - e
        out = v_norm * x_norm * np.exp(-.5*np.squeeze(exp, (-2, -1)))

        out *= self.rbf_var
        if X2 is None:
            out = (out + out.T) / 2
            out.flat[::len(X)+1] = self.rbf_var + self.white_var
        #print("To compute matrix took me:", time.time() - start)
        assert out.shape == (len(X), len(X) if X2 is None else len(X2))
        return out


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

    M = np.eye(test_df.shape[1]) / 1000
    x = uncertain_point(test_df.values[0], m, 0.99, M)
    v = uncertain_point(test_df.values[1], m, 0.99, M)

    print(rbf_uncertain(x, v, M))
    print(rbf_uncertain(v, x, M))
