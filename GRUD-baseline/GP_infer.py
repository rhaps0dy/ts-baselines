from __future__ import print_function, division

import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import pickle_utils as pu
import GPflow
import time
import tensorflow as tf
import sys

def build_gp(X, y, sample_variance, meta_variance, M=3000):
    dims = X.shape[1]
    rbf = GPflow.kernels.RBF(dims, variance=1, lengthscales=[X[:,1].mean(), 100], ARD=True)
    rbf.lengthscales.fixed = True
    rbf.variance.fixed = True
    white = GPflow.kernels.White(dims, variance=10)
    #white.variance.fixed = True

    mask = (np.random.random(X.shape[0]) < (M/X.shape[0]))
    m = GPflow.sgpr.SGPR(X, y, kern=rbf+white, Z=X[mask])
    m.Z.fixed = True

    def logger(x):
        if (logger.i % 10) == 0:
            print(x, m._objective(x)[0])
        logger.i+=1
    logger.i = 0

    #m.kern.variance.prior = GPflow.priors.Gaussian(
    #    mu=sample_variance, var=max(meta_variance, 1e-3))
    #m.kern.variance = sample_variance
    #import code; code.interact(local=locals())
    #m.kern.variance.fixed = True
    #m.optimize(method=tf.train.AdamOptimizer(), callback=logger)
    return m

rng = np.random.RandomState(0)

X, z = pu.load(sys.argv[1])
X = np.array(X, dtype=np.float64)
z = np.array(z, dtype=np.float64)

samples = np.concatenate([[X[0, 0]], z])
mean = np.mean(samples)
sample_variance = np.var(samples, ddof=1, dtype=np.float)
# Assuming the input forms a normal distribution, the variance of the sample
# variance is:
meta_variance = 2*sample_variance**4/(len(samples)-1)
print("Mean:", mean, "var:", sample_variance, "meta_variance:", meta_variance)

z = z[:,None]

z -= mean
X[:,0] -= mean

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(X[:,0], X[:,1], z[:,0], c='r', marker='o')
#
m = build_gp(X, z, sample_variance, meta_variance)

pX = np.array(list(zip(*map(np.ravel, np.meshgrid(
    np.linspace(X[:,0].min(), X[:,0].max(), 20),
    np.linspace(X[:,1].min(), X[:,1].max(), 20))))), dtype=np.float64)

pY, pYv = m.predict_y(pX)
pZ, _ = m.predict_y(m.Z.value)
pu.dump((pY, pYv, m.Z.value, pZ), "{:s}_pY.pkl.gz".format(sys.argv[1][:-7]))

#plt.show()
