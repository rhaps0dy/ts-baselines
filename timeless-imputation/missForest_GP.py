import numpy as np
import pandas as pd
import tensorflow as tf
import GPy


class GPRegression:
    def __init__(self, n_features):
        pass

    @staticmethod
    def make_kernel(X):
        n_features = X.shape[1]
        kern = GPy.kern.Matern32(n_features, ARD=False)

    def fit(self, X, y, max_iters=1000, num_inducing=300):
        kern = self.make_kernel(X)
        n_features = X.shape[1]
        y = y.values[:, np.newaxis]
        if len(X) <= num_inducing:
            self.m = GPy.models.GPRegression(X.values, y, kern)
        else:
            self.m = GPy.models.SparseGPRegression(X.values, y, kernel=kern,
                                                   num_inducing=num_inducing)
        print(self.m.optimize('bfgs', max_iters=max_iters))

    def predict(self, X):
        return self.m.predict(X.values)[0]


class GPClassification(GPRegression):
    def fit(self, X, y, max_iters=1000, num_inducing=300):
        kern = self.make_kernel(X)
        y = y.values[:, np.newaxis]
        n_classes = y.max()+1
        gp_dense = len(X) <= num_inducing

        if n_classes == 2:
            if gp_dense:
                C = GPy.models.GPClassification
            else:
                C = GPy.models.SparseGPClassification
        else:
            raise NotImplementedError, "Multiclass classification"
            if gp_dense:
                C = GPy.models.OneVsAllClassification
            else:
                C = GPy.models.OneVsAllSparseClassification

        if gp_dense:
            self.m = C(X.values, y, kernel=kern)
        else:
            self.m = C(X.values, y, kernel=kern, num_inducing=num_inducing)
        print(self.m.optimize('bfgs', max_iters=max_iters))

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        return self.m.predict(X.values)[0]


classifier = GPClassification
regressor = GPRegression


if __name__ == '__main__':
    import missForest
    import datasets

    _ds = datasets.datasets()
    dsets = dict((x, _ds[x]) for x in ["BostonHousing",
        # "Soybean", "BreastCancer", "Servo", "Ionosphere"
    ])
    baseline = datasets.benchmark({
        'GP_vanilla_300': datasets.memoize(lambda d, full_data: \
                                           missForest.impute(
            d, full_data, sequential=True, print_progress=True,
            predictors=(GPClassification, GPRegression),
            use_previous_prediction=True)),
    }, dsets, do_not_compute=False)
