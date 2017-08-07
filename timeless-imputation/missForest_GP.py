import numpy as np
import pandas as pd
import tensorflow as tf
import GPy
import mog_rbf


class GPRegression:
    def __init__(self, n_features, mog, ARD=False):
        print("Your variety of GP is:", self.__class__)
        self.ARD = ARD
        self.mog = mog

    def make_kernel(self, X):
        n_features = X.shape[1]
        kern = GPy.kern.RBF(n_features, ARD=self.ARD) + GPy.kern.White(n_features)
        return kern

    def fit(self, X, X_var, y, max_iters=1000, num_inducing=300, optimize=True):
        kern = self.make_kernel(X)
        n_features = X.shape[1]
        y = y.values[:, np.newaxis]
        self.m = GPy.models.GPRegression(X=X.values, Y=y, kernel=kern)
        if optimize:
            print(self.m.optimize('bfgs', max_iters=max_iters))

    def predict(self, X, X_var):
        return self.m.predict(X.values)

class VariationalGPRegression(GPRegression):
    def fit(self, X, X_var, y, max_iters=1000, optimize=False):
        kern = self.make_kernel(X)
        n_features = X.shape[1]
        y = y.values[:, np.newaxis]
        self.m = GPy.models.SparseGPRegression(X=X.values, Y=y, kernel=kern,
                                               Z=X.values,
                                               X_variance=X_var.values)
        if optimize:
            print(self.m.optimize('bfgs', max_iters=max_iters))

class UncertainGPRegression(GPRegression):
    Model = GPy.models.GPRegression
    def fit(self, X, X_var, y, max_iters=1000, optimize=False):
        if self.mog is None:
            kern = mog_rbf.UncertainGaussianRBFWhite(X.shape[1], None)
            inputs = np.concatenate([X.values, X_var.values], axis=1)
        else:
            kern = mog_rbf.UncertainMoGRBFWhite(X.shape[1], self.mog)
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
    def fit(self, X, X_var, y, max_iters=1000, num_inducing=300, optimize=True):
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
            print(self.m.optimize('bfgs', max_iters=max_iters))

    def predict(self, X, X_var):
        raise NotImplementedError

    def predict_proba(self, X, X_var):
        return self.m.predict(X.values)

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
            print(self.m.optimize('bfgs', max_iters=max_iters))

if __name__ == '__main__':
    import missForest
    import datasets
    import missing_bayesian_mixture as mbm

    _ds = datasets.datasets()
    dsets = dict((x, _ds[x]) for x in ["BostonHousing" # "Ionosphere", "BostonHousing",
        # "Soybean", "BreastCancer", "Servo", "Ionosphere"
    ])
    baseline = datasets.benchmark({
        #'GP_ARD': lambda log_path, d, full_data: missForest.impute(
        #    log_path, d, full_data, sequential=False, print_progress=True,
        #    predictors=(lambda *args: GPClassification(*args, ARD=True),
        #                lambda *args: GPRegression(*args, ARD=True)),
        #    use_previous_prediction=False),
        #'GP': lambda log_path, d, full_data: missForest.impute(
        #    log_path, d, full_data, sequential=False, print_progress=True,
        #    predictors=(GPClassification, GPRegression),
        #    use_previous_prediction=False),
        #'GP_unopt': lambda log_path, d, full_data: missForest.impute(
        #    log_path, d, full_data, sequential=False, print_progress=True,
        #    predictors=(GPClassification, GPRegression),
        #    use_previous_prediction=False, optimize_gp=False),
        'GP_kern_mog': lambda log_path, d, full_data: missForest.impute(
            log_path, d, full_data, sequential=False, print_progress=True,
            predictors=(UncertainGPClassification, UncertainGPRegression),
            impute_name_replace=('GP_kern_mog', 'GMM'),
            initial_impute=mbm.mf_initial_impute,
            use_previous_prediction=False, optimize_gp=False),
        'GP_kern_uncertain': lambda log_path, d, full_data: missForest.impute(
            log_path, d, full_data, sequential=False, print_progress=True,
            predictors=(UncertainGPClassification, UncertainGPRegression),
            use_previous_prediction=False, optimize_gp=False),
        #'GP_uncertain': lambda log_path, d, full_data: missForest.impute(
        #    log_path, d, full_data, sequential=False, print_progress=True,
        #    predictors=(VariationalGPClassification, VariationalGPRegression),
        #    use_previous_prediction=False, optimize_gp=False),
    }, dsets, do_not_compute=False)
