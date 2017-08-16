import numpy as np
import pandas as pd
import tensorflow as tf
import GPy
import mog_rbf
import knn_kernel


class GPRegression:
    def __init__(self, n_features, mog, complete_X, n_neighbours, knn_type,
                 ARD=True, params=None):
        self.ARD = ARD
        self.mog = mog
        self.complete_X = complete_X[~np.all(np.isnan(complete_X), axis=1)]
        self.n_neighbours = n_neighbours
        self.knn_type = knn_type
        self.params = params

    def make_kernel(self, X):
        n_features = X.shape[1]
        kern = GPy.kern.Matern32(n_features, ARD=self.ARD) + GPy.kern.White(n_features)
        return kern

    def fit(self, X, X_var, y, max_iters=1000, optimize=True):
        kern = self.make_kernel(X)
        n_features = X.shape[1]
        y = y.values[:, np.newaxis]
        self.m = GPy.models.GPRegression(X=X.values, Y=y, kernel=kern)
        if optimize:
            self.m.optimize('bfgs', max_iters=max_iters)

    def predict(self, X, X_var):
        return self.m.predict(X.values)

class KNNKernelMixin:
    def make_kernel(self, X):
        if np.any(np.isnan(X)):
            return knn_kernel.RBFWhiteKNNCheating(self.complete_X.values,
                                                  n_neighbours=self.n_neighbours,
                                                  knn_type=self.knn_type,
                                                  ARD=self.ARD)
        else:
            return knn_kernel.RBFWhiteKNNCheating(self.complete_X.values,
                                                n_neighbours=1, ARD=self.ARD)

    def fit(self, X, X_var, y, max_iters=1000, optimize=True):
        kern = self.make_kernel(X)
        y = y.values[:, np.newaxis]
        X = X.values
        missing_rows = np.all(np.isnan(X), axis=1)
        self.m = GPy.models.GPRegression(X=X[~missing_rows],
                                         Y=y[~missing_rows], kernel=kern)
        if optimize:
            self.m.optimize('bfgs', max_iters=max_iters)

    def predict(self, X, X_var):
        missing_rows = np.all(np.isnan(X.values), axis=1)
        mean, var = self.m.predict(X.values)
        # Return the prior for missing rows
        mean[missing_rows] = 0.
        var[missing_rows] = 1.
        return mean, var

class KNNGPRegression(KNNKernelMixin, GPRegression):
    pass

class VariationalGPRegression(GPRegression):
    def fit(self, X, X_var, y, max_iters=1000, optimize=False):
        kern = self.make_kernel(X)
        n_features = X.shape[1]
        y = y.values[:, np.newaxis]
        self.m = GPy.models.SparseGPRegression(X=X.values, Y=y, kernel=kern,
                                               Z=X.values,
                                               X_variance=X_var.values)
        if optimize:
            self.m.optimize('bfgs', max_iters=max_iters)

class UncertainGPRegression(GPRegression):
    Model = GPy.models.GPRegression
    def fit(self, X, X_var, y, max_iters=1000, optimize=False):
        if self.mog is None:
            kern = mog_rbf.UncertainGaussianRBFWhite(X.shape[1], None)
            inputs = np.concatenate([X.values, X_var.values], axis=1)
        else:
            #import mog_rbf_tf
            if self.params is not None:
                kern = mog_rbf.UncertainMoGRBFWhite(X.shape[1], self.mog)
            else:
                kern = mog_rbf.UncertainMoGRBFWhite(
                    X.shape[1], self.mog,
                    white_var=self.params['white_variance'],
                    rbf_var=self.params['rbf_variance'],
                    lengthscale=self.params['rbf_lengthscale'])
            inputs = X.values.copy()
            inputs[X_var.values != 0.0] = np.nan
        self.m = self.Model(X=inputs, Y=y.values[:, np.newaxis], kernel=kern)

    def predict(self, X, X_var):
        if self.mog is None:
            inputs = np.concatenate([X.values, X_var.values], axis=1)
        else:
            inputs = X.values.copy()
            inputs[X_var.values != 0.0] = np.nan
        return self.m.predict(inputs)


class UncertainGPClassification(UncertainGPRegression):
    Model = GPy.models.GPClassification
    def predict_proba(self, X, X_var):
        return self.predict(X, X_var)

class GPClassification(GPRegression):
    def fit(self, X, X_var, y, max_iters=1000, optimize=True):
        kern = self.make_kernel(X)
        y = y.values[:, np.newaxis]
        n_classes = y.max()+1

        if n_classes == 2:
            C = GPy.models.GPClassification
        else:
            raise NotImplementedError("Multiclass classification")
            if gp_dense:
                C = GPy.models.OneVsAllClassification
            else:
                C = GPy.models.OneVsAllSparseClassification

        self.m = C(X=X.values, Y=y, kernel=kern)
        if optimize:
            self.m.optimize('bfgs', max_iters=max_iters)

    def predict(self, X, X_var):
        raise NotImplementedError

    def predict_proba(self, X, X_var):
        return self.m.predict(X.values)

class KNNGPClassification(KNNKernelMixin, GPClassification):
    pass


class VariationalGPClassification(GPClassification):
    def fit(self, X, X_var, y, max_iters=1000, optimize=False):
        kern = self.make_kernel(X)
        n_features = X.shape[1]
        y = y.values[:, np.newaxis]
        n_classes = y.max()+1
        if n_classes == 2:
            C = GPy.models.SparseGPClassificationUncertainInput
        else:
            raise NotImplementedError("Multiclass classification")
        self.m = C(X=X.values, Y=y, kernel=kern, Z=X.values,
                   X_variance=X_var.values)
        if optimize:
            self.m.optimize('bfgs', max_iters=max_iters)

if __name__ == '__main__':
    import missForest
    import datasets
    import missing_bayesian_mixture as mbm

    _ds = datasets.datasets()
    dsets = dict((x, _ds[x]) for x in ["BostonHousing", "Ionosphere", # "Servo",
        # "Soybean", "BreastCancer", "Servo", "Ionosphere"
    ])
    for i in range(10):
        baseline = datasets.benchmark({
            ('GP_KNN_meanimp_7_iter{:02d}'
             .format(i)): lambda log_path, d, full_data: missForest.impute(
                log_path, d, full_data, sequential=False, print_progress=True,
                predictors=(KNNGPClassification, KNNGPRegression),
                optimize_gp=True, use_previous_prediction=False,
                 max_iterations=1, n_neighbours=7, knn_type='mean'),
            ('GP_KNN_kmean_7_iter{:02d}'
             .format(i)): lambda log_path, d, full_data: missForest.impute(
                log_path, d, full_data, sequential=False, print_progress=True,
                predictors=(KNNGPClassification, KNNGPRegression),
                optimize_gp=True, use_previous_prediction=False,
                max_iterations=1, n_neighbours=7, knn_type='kernel_weighted_mean'),
            ('GP_KNN_kernel_7_iter{:02d}'
             .format(i)): lambda log_path, d, full_data: missForest.impute(
                log_path, d, full_data, sequential=False, print_progress=True,
                predictors=(KNNGPClassification, KNNGPRegression),
                optimize_gp=True, use_previous_prediction=False,
                max_iterations=1, n_neighbours=7, knn_type='kernel_avg'),
            ('GP_KNN_meanimp_5_iter{:02d}'
             .format(i)): lambda log_path, d, full_data: missForest.impute(
                log_path, d, full_data, sequential=False, print_progress=True,
                predictors=(KNNGPClassification, KNNGPRegression),
                optimize_gp=True, use_previous_prediction=False,
                max_iterations=1, n_neighbours=5, knn_type='mean'),
            ('GP_KNN_kmean_5_iter{:02d}'
             .format(i)): lambda log_path, d, full_data: missForest.impute(
                log_path, d, full_data, sequential=False, print_progress=True,
                predictors=(KNNGPClassification, KNNGPRegression),
                optimize_gp=True, use_previous_prediction=False,
                max_iterations=1, n_neighbours=5, knn_type='kernel_weighted_mean'),
            ('GP_KNN_kernel_5_iter{:02d}'
             .format(i)): lambda log_path, d, full_data: missForest.impute(
                log_path, d, full_data, sequential=False, print_progress=True,
                predictors=(KNNGPClassification, KNNGPRegression),
                optimize_gp=True, use_previous_prediction=False,
                max_iterations=1, n_neighbours=5, knn_type='kernel_avg'),
        }, dsets, do_not_compute=False)
        print(baseline)
