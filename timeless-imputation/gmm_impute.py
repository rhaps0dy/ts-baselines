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
        d[attr] = getattr(m, attr+'_')[mask, ...]
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

    diffs = inputs[_mask] - means[:, _mask]
    right_diffs = np.expand_dims(diffs, axis=2)
    left_diffs = np.expand_dims(diffs, axis=1)
    _covs = mask_matrix(covs, _mask, _mask)
    exponent = left_diffs @ np.linalg.inv(_covs) @ right_diffs

    exponent = -.5 * np.squeeze(exponent, axis=[1, 2])
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
            out[:, j, :] = matrix[:, i, m2]
            j += 1
    return out


def conditional_mog(m, inp, mask, cutoff=1.0):
    if np.all(mask):
        return {'means': m['means'],
                'covariances': m['covariances'],
                'weights': m['weights']}
    if np.all(~mask):
        n_gaussians = 1
        ws = np.zeros([n_gaussians], dtype=inp.dtype)
        ws[0] = 1
        return {'means': np.zeros([n_gaussians, 0], dtype=inp.dtype),
                'covariances': np.zeros([n_gaussians, 0, 0], dtype=inp.dtype),
                'weights': ws}
    # return {'means': np.zeros([1, np.sum(mask)], dtype=inp.dtype),
    #         'covariances': np.eye(np.sum(mask))[np.newaxis, :, :],
    #         'weights': np.ones([1], dtype=inp.dtype)}

    # $\tau$ in "Imputation through finite Gaussian mixture models" (Di
    # Zio et al., 2007)
    w = m['weights'] * gaussian_pdf(inp, m['means'], m['covariances'])

    normaliser = np.sum(w)
    if normaliser == 0.0:
        w = m['weights']
    else:
        w /= normaliser

    if cutoff < 1.0:
        w_i = np.argsort(w)[::-1]
        m_keep = np.zeros_like(w, dtype=np.bool)
        m_keep[w_i] = np.cumsum(w[w_i]) < cutoff
        # Move one forward, so that the last True item is just above the cutoff
        m_keep[w_i] = np.concatenate([[True], m_keep[w_i[:-1]]])
        # Renormalise cluster weights that are left
        weights = w[m_keep]
        del w
        means = m['means'][m_keep, :]
        covariances = m['covariances'][m_keep, :, :]

        normaliser = np.sum(weights)
        assert normaliser > 0.0
        weights /= normaliser
    else:
        weights = w
        means = m['means']
        covariances = m['covariances']

    K_12 = mask_matrix(covariances, mask, ~mask)
    K_22 = mask_matrix(covariances, ~mask, ~mask)
    K_22__1 = np.linalg.inv(K_22)
    K_1222 = K_12 @ K_22__1
    K_21 = mask_matrix(covariances, ~mask, mask)
    K_11 = mask_matrix(covariances, mask, mask)

    diff = np.expand_dims(inp[~mask] - means[:, ~mask], axis=2)
    return {
        'means': means[:, mask] + np.squeeze(K_1222 @ diff, axis=2),
        'covariances': K_11 - K_1222 @ K_21,
        'weights': weights}

def single_gaussian_moment_matching(d):
    ms = d['means'] * d['weights'][:, np.newaxis]
    mean = np.sum(ms, axis=0)
    # Take only the diagonal of covariances
    # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
    variances = np.sum(d['covariances'] * np.eye(d['covariances'].shape[1]), axis=2)
    var_mix = np.sum(variances * d['weights'][:, np.newaxis], axis=0)
    var = var_mix + np.sum(ms * d['means'], axis=0) - mean**2
    return mean, var


def _gmm_impute(m, inputs, n_impute, sample_impute=False):
    "Impute inputs using Gaussian Mixture Model m"
    if sample_impute:
        outputs = np.stack([inputs] * n_impute, axis=0)
    else:
        outputs = np.expand_dims(inputs.copy(), axis=0)
    outputs_var = np.zeros_like(inputs)
    for i, (inp, mask) in enumerate(zip(inputs, np.isnan(inputs))):
        if not np.any(mask):
            continue

        d = conditional_mog(m, inp, mask, cutoff=1.0)
        outputs[:, i, ~mask] = inp[~mask]

        mean, var = single_gaussian_moment_matching(d)
        outputs[:, i, mask] = mean
        outputs_var[i, mask] = var

        if sample_impute:
            _mask = np.zeros_like(mask, dtype=np.bool)
            _mask[sample_impute] = True
            outputs[:, i, mask & _mask] = samples_from_mixture(d, n_impute)[:, _mask[mask]]
    return outputs, outputs_var

def imputed_log_likelihood(m, inputs, targets, num_keys):
    log_likelihood = 0.
    num_key_mask = np.zeros([inputs.shape[1]], dtype=np.bool)
    num_key_mask[num_keys] = True
    all_missing_inputs = np.isnan(inputs)
    for inp, mask, tg in zip(inputs, all_missing_inputs, targets):
        if not np.any(mask & num_key_mask):
            continue
        d = conditional_mog(m, inp, mask, cutoff=1.0)

        means = d['means'][:, num_key_mask[mask]]
        covariances = (d['covariances']
                       [:, num_key_mask[mask], :][:, :, num_key_mask[mask]])

        diffs = tg[mask & num_key_mask] - means
        n_features = np.sum(mask, axis=0, keepdims=True)
        right_diffs = np.expand_dims(diffs, axis=2)
        left_diffs = np.expand_dims(diffs, axis=1)
        exponent = left_diffs @ np.linalg.inv(covariances) @ right_diffs

        exponent = -.5 * np.squeeze(exponent, axis=[1, 2])
        divider = np.linalg.det(covariances)**(-.5)
        likelihood = np.sum(d['weights'] * divider * np.exp(exponent), axis=0)
        assert len(likelihood.shape) == 0
        log_likelihood += np.log(np.clip(likelihood, np.finfo(np.float64).tiny, np.inf))

    log_likelihood -= .5 * np.sum(all_missing_inputs) * np.log(2*np.pi)
    return log_likelihood
