"""Impute data using Gaussian Mixture Models"""

import numpy as np

def prune_infinite_gmm(X, m):
    """ Discards clusters that do not explain anything in `X` from the
    (Bayesian)GaussianMixture `m` """
    Y = m.predict(X)
    mask = np.zeros(len(m.weights_), dtype=np.bool)
    for i in set(Y):
        mask[i] = True
    d = {}
    for attr in ['weights', 'means', 'covariances']:
        d[attr] = getattr(m, attr+'_')[mask,...]
    # Proportionally share the weight of the clusters left out
    shared_weight = np.sum(m.weights_[~mask])
    d['weights'] /= (1-shared_weight)
    return d

def samples_from_mixture(m, n):
    """Draw `n` samples from the GMM `m`."""
    i = np.random.choice(np.arange(len(m['weights'])), p=m['weights'], size=[n])
    L_covs = np.linalg.cholesky(m['covariances'])[i]
    means = m['means'][i]
    outputs = np.random.randn(*means.shape, 1)
    return means + np.squeeze(np.matmul(L_covs, outputs), axis=2)

def gaussian_pdf(inputs, means, covs):
    """Compute the gaussian PDF for `inputs`, and
    each row of `means` and `covs`."""
    _mask = ~np.isnan(inputs)
    n_features = np.sum(_mask, axis=0, keepdims=True)

    diffs = inputs[_mask] - means[:,_mask]
    right_diffs = np.expand_dims(diffs, axis=2)
    left_diffs = np.expand_dims(diffs, axis=1)
    _covs = mask_matrix(covs, _mask, _mask)
    exponent = left_diffs @ np.linalg.inv(_covs) @ right_diffs

    exponent = -.5 * np.squeeze(exponent, axis=[2,3])
    divider = ((2 * np.pi)**n_features * np.linalg.det(_covs))**(-.5)
    g_pdf = divider * np.exp(exponent)
    return g_pdf

def mask_matrix(matrix, m1, m2):
    "Use two mask indices on a matrix"
    d1 = np.sum(m1)
    d2 = np.sum(m2)
    out = np.empty(shape=[len(matrix), d1, d2])
    j = 0
    for i, b in enumerate(m1):
        if b:
            out[:,j,:] = matrix[:,i,m2]
            j += 1
    return out


def _gmm_impute(m, inputs, n_impute=100):
    "Impute inputs using Gaussian Mixture Model m"
    outputs = np.empty(shape=[n_impute]+list(inputs.shape),
                       dtype=np.float)
    for i, inp in enumerate(inputs):
        mask = np.isnan(inp)
        if not np.any(mask):
            outputs[:,i,:] = inp
            continue
        d = {}
        if not np.any(~mask):
            d['means'] = m['means']
            d['covariances'] = m['covariances']
            d['weights'] = m['weights']
        else:
            g_pdf = gaussian_pdf(inp, m['means'], m['covariances'])
            # $\tau$ in "Imputation through finite Gaussian mixture models" (Di
            # Zio et al., 2007)
            d['weights'] = m['weights'] * g_pdf
            d['weights'] /= np.sum(d['weights'], axis=0, keepdims=True)

            K_12 = mask_matrix(m['covariances'],mask,~mask)
            K_22 = mask_matrix(m['covariances'],~mask,~mask)
            K_22__1 = np.linalg.inv(K_22)
            K_1222 = K_12 @ K_22__1
            K_21 = mask_matrix(m['covariances'],~mask,mask)
            K_11 = mask_matrix(m['covariances'],mask,mask)

            diff = np.expand_dims(inp[~mask] - m['means'][:,~mask], axis=2)
            d['means'] = m['means'][:,mask] + np.squeeze(K_1222 @ diff, axis=2)
            d['covariances'] = K_11 - K_1222 @ K_21

            outputs[:,i,~mask] = inp[~mask]
        outputs[:,i,mask] = samples_from_mixture(d, n_impute)
        #outputs[:,i,mask] = [np.mean(d['means'], axis=0)] * n_impute
    return outputs
