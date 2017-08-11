import numpy as np
import GPy
from paramz.caching import Cache_this
import time


class RBFWhiteKNNCheating(GPy.kern.src.kern.CombinationKernel):
    def __init__(self, complete_dset, n_neighbours=5, white_var=1., rbf_var=1.,
                 knn_type='kernel_avg', name='knn', **kwargs):
        rbf = GPy.kern.RBF(complete_dset.shape[1], variance=rbf_var,
                           name='rbf', **kwargs)
        white = GPy.kern.White(complete_dset.shape[1], variance=white_var,
                               name='white')
        super(RBFWhiteKNNCheating, self).__init__([rbf, white], name)
        self.n_neighbours = n_neighbours
        self.complete_dset = complete_dset
        self.reset_times()

        assert knn_type in {'kernel_avg', 'kernel_weighted_mean', 'mean'}
        self.knn_type = knn_type

    def reset_times(self):
        self.neighbours_time = 0.

    def print_times(self):
        print("neighbours time:", self.neighbours_time, "seconds")

    @Cache_this(limit=3, ignore_args=())
    def neighbours(self, X):
        start_time = time.time()
        diff = X[:, np.newaxis, :] - self.complete_dset[np.newaxis, :, :]
        missing = np.isnan(diff)
        diff[missing] = 0
        if len(self.rbf.lengthscale) == 1:
            dist = np.sum(diff**2, axis=2) / self.rbf.lengthscale
        else:
            dist = np.sum((diff/self.rbf.lengthscale)**2, axis=2)
        correct = missing.shape[2] / np.sum(~missing, axis=2)
        # If some rows are missing, it doesn't matter what value they get.
        # The error gets corrected later
        correct[np.isnan(correct)] = 0.

        bad_kernel_dist = np.exp(-.5 * dist * correct)
        # The diagonal gets sorted to the end, it has the maximum kernel
        # distance
        neighbours_i_all = np.argsort(bad_kernel_dist, axis=-1)
        neighbours_i = neighbours_i_all[..., -1-self.n_neighbours:-1]
        self.neighbours_time += time.time() - start_time
        # used to return just the next line
        neigh_values = self.complete_dset[neighbours_i]
        neigh_mean = np.nanmean(neigh_values, axis=1, keepdims=True)

        mean_nan = np.isnan(neigh_mean)
        mean_nan_rows = np.any(mean_nan, axis=(1, 2))
        j = 1
        while np.any(mean_nan_rows):
            end_i = -1 - j*self.n_neighbours
            if end_i < -neighbours_i_all.shape[1]:
                print("WARNING: no more neighbours")
                neigh_mean[mean_nan] = 0.
                break
            for (i,) in zip(*np.nonzero(mean_nan_rows)):
                next_neighbours = neighbours_i_all[
                    i, end_i-self.n_neighbours:end_i]
                mask = mean_nan[i, 0]
                neigh_mean[i, 0, mask] = np.nanmean(
                    self.complete_dset[next_neighbours], axis=0
                    )[mask]
            j += 1
            mean_nan = np.isnan(neigh_mean)
            mean_nan_rows = np.any(mean_nan, axis=(1, 2))
        if self.knn_type == 'kernel_avg':
            out = neigh_values
        elif self.knn_type == 'kernel_weighted_mean':
            try:
                out = np.nanmean(
                    bad_kernel_dist.take(neighbours_i)[:, :, np.newaxis]
                    * neigh_values, axis=1, keepdims=True)
            except IndexError:
                import pdb
                pdb.set_trace()
        elif self.knn_type == 'mean':
            out = np.nanmean(neigh_values, axis=1, keepdims=True)
        return np.where(np.isnan(out), neigh_mean, out)

    @Cache_this(limit=3, ignore_args=())
    def neighbour_average(self, fun, X, X2=None):
        neigh_X = self.neighbours(X).copy()
        observed = ~np.isnan(X)
        np.transpose(neigh_X, (1, 0, 2))[:, observed] = X[observed]

        if X2 is None:
            neigh_X2 = neigh_X
        else:
            neigh_X2 = self.neighbours(X2).copy()
            observed_X2 = ~np.isnan(X2)
            np.transpose(neigh_X2, (1, 0, 2))[:, observed_X2] = X2[observed_X2]

        out = None
        for i in range(neigh_X.shape[1]):
            for j in range(neigh_X2.shape[1]):
                a = fun(neigh_X[:, i, :], neigh_X2[:, j, :])
                if out is None:
                    out = a
                else:
                    out += a
        return out / float(neigh_X.shape[1] * neigh_X2.shape[1])

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def K(self, X, X2=None, which_parts=None):
        k = self.neighbour_average(self.rbf.K, X, X2)
        if X2 is None:
            k.flat[::k.shape[0]+1] += self.white.Kdiag(X)
        return k

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def Kdiag(self, X, which_parts=None):
        return self.rbf.Kdiag(X) + self.white.Kdiag(X)

    def update_gradients_full(self, dL_dK, X, X2=None):
        def concatenated_gradients(_X, _X2):
            self.rbf.update_gradients_full(dL_dK, _X, _X2)
            return np.concatenate(
                [self.rbf.variance.gradient,
                 self.rbf.lengthscale.gradient], axis=0)
        out = self.neighbour_average(concatenated_gradients, X, X2)
        self.rbf.variance.gradient = out[0]
        self.rbf.lengthscale.gradient = out[1:]
        if X2 is None:
            self.white.variance.gradient = np.trace(dL_dK)
        else:
            self.white.variance.gradient = 0.0

    def update_gradients_diag(self, dL_dKdiag, X):
        self.rbf.lengthscale.gradient[...] = 0.0
        self.white.variance.gradient = \
            self.rbf.variance.gradient = np.sum(dL_dKdiag)
