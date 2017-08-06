import numpy as np
import gmm_impute as gmm
import unittest

def tp(x):
    return np.swapaxes(x, -1, -2)

def uncertain_point_curried(m, cutoff=0.99):
    def f(inp, M):
        mis = np.isnan(inp)
        d = gmm.conditional_mog(m, inp, mis, cutoff=cutoff)
        d['mis'] = mis
        d['mu_obs'] = inp[~mis][np.newaxis, :, np.newaxis]
        p = np.linalg.inv(d['covariances'])
        d['precisions'] = p
        A = d['precisions'] + np.diag(M[mis])
        d['Ainv'] = np.linalg.inv(A)
        d['sqrt_det'] = (np.linalg.det(d['covariances'])
                         * np.linalg.det(A))**(-.5)

        d['means'] = d['means'][:, :, np.newaxis]
        d['sig_mu'] = p @ d['means']
        d['mu_sig_mu'] = tp(d['means']) @ d['sig_mu']
        d['neg_Ainv_a'] = d['Ainv'] @ d['sig_mu']
        d['a_Ainv_a'] = tp(d['sig_mu']) @ d['neg_Ainv_a']
        return d
    return f


def rbf_uncertain(x, v, M=None, std_f=1., std_n=1.):
    if x['id'] == v['id']:
        return std_n
    n_dims = x['mis'].shape[-1]
    if M is None:
        M = np.ones(n_dims, dtype=x['means'].dtype)
    M = np.diag(M)[np.newaxis, np.newaxis, :, :]
    assert M.shape == (1, 1, n_dims, n_dims), \
        "M not vector representing a diagonal matrix"

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
    print(exp)

    return std_f * np.sum(ww * np.exp(-.5*exp))

class TestFunction(unittest.TestCase):
    N = 30
    def test_symmetric(self):
        np.random.rand(2*N, )

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
    x = uncertain_point_curried(m)(test_df.values[0], M)
    x['id'] = 0
    v = uncertain_point_curried(m)(test_df.values[1], M)
    v['id'] = 1

    print(rbf_uncertain(x, v, M))
    print(rbf_uncertain(v, x, M))
