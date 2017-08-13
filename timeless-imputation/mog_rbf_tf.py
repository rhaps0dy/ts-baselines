import tensorflow as tf
from add_variable_scope import add_variable_scope

from GPy.kern.src.kern import Kern
from GPy.core.parameterization import Param
from paramz.caching import Cache_this
import numpy as np


def tp(t):
    "Swap the last two axes of an array"
    l = list(range(len(t.get_shape())))
    a = l[-1]
    l[-1] = l[-2]
    l[-2] = a
    return tf.transpose(t, l)

# Tested cursorily against numpy implementation
@add_variable_scope(name="gaussian_pdf")
def tf_gaussian_pdf(diffs, right_diffs, covariances, inverse_covariances):
    """Compute the gaussian PDF for each row of `inputs`, and
    each element of GMM."""
    left_diffs = tf.expand_dims(diffs, axis=1)
    exponent = left_diffs @ inverse_covariances @ right_diffs

    exponent = -0.5 * tf.squeeze(exponent, axis=[1, 2])
    dets = tf.matrix_determinant(covariances)
    divider = tf.pow(tf.constant(2 * np.pi, dtype=diffs.dtype),
                     tf.cast(tf.shape(diffs)[1], dtype=diffs.dtype))
    rsqrt_div = tf.rsqrt(divider * dets)
    g_pdf = tf.exp(exponent) * rsqrt_div
    return g_pdf


# Tested cursorily against numpy implementation
@add_variable_scope(name="tf_conditional_mog")
def tf_conditional_mog(_inputs, GMM, cutoff=1.0):
    """Calculates the conditional distribution of _inputs using the Gaussian
    Mixture Model `GMM`"""
    inputs, mask = _inputs

    tf_float = inputs.dtype
    n_missing = tf.reduce_sum(tf.to_int32(mask), axis=0)
    not_n_missing = int(inputs.get_shape()[0]) - n_missing
    assert len(n_missing.get_shape()) == 0, "n_missing is a scalar"

    n_gaussians = int(GMM['weights'].get_shape()[0])
    n_variables = int(GMM['means'].get_shape()[1])

    mis_1d = tf.where(mask)
    obs_1d = tf.where(~mask)
    n_mis = tf.shape(mis_1d)[0]
    n_obs = tf.shape(obs_1d)[0]

    main_index_source = tf.constant(np.transpose(np.stack(
        [np.stack([np.arange(n_gaussians)] * n_variables)] * n_variables), (2, 0, 1)),
                                    dtype=tf.int64)

    mis_tile = tf.tile(tf.expand_dims(mis_1d, 0), [n_gaussians, 1, n_variables])
    obs_tile = tf.tile(tf.expand_dims(obs_1d, 0), [n_gaussians, 1, n_variables])

    mis_mis_tile = mis_tile[:, :, :n_mis]
    mis_obs_tile = mis_tile[:, :, :n_obs]
    obs_mis_tile = obs_tile[:, :, :n_mis]
    obs_obs_tile = obs_tile[:, :, :n_obs]

    mis_mis = tf.stack([main_index_source[:, :n_mis, :n_mis],
                        mis_mis_tile, tp(mis_mis_tile)], axis=3)
    mis_obs = tf.stack([main_index_source[:, :n_mis, :n_obs],
                        mis_obs_tile, tp(obs_mis_tile)], axis=3)
    obs_mis = tf.stack([main_index_source[:, :n_obs, :n_mis],
                        obs_mis_tile, tp(mis_obs_tile)], axis=3)
    obs_obs = tf.stack([main_index_source[:, :n_obs, :n_obs],
                        obs_obs_tile, tp(obs_obs_tile)], axis=3)

    mis = tf.stack(
        [main_index_source[:, :n_mis, 0], mis_tile[:, :, 0]], axis=2)
    obs = tf.stack(
        [main_index_source[:, :n_obs, 0], obs_tile[:, :, 0]], axis=2)

    @add_variable_scope(name="conditional_dist")
    def conditional_dist():
        # For the moment we don't prune any gaussian
        K_22 = tf.gather_nd(GMM['covariances'], obs_obs, name="K_22")
        K_22__1 = tf.matrix_inverse(K_22)
        known_inputs_diff = (tf.boolean_mask(inputs, ~mask)
                             - tf.gather_nd(GMM['means'], obs))
        right_diff = tf.expand_dims(known_inputs_diff, axis=2)
        g_pdf = tf_gaussian_pdf(known_inputs_diff, right_diff, K_22, K_22__1)
        unnorm_new_weights = GMM['weights'] * g_pdf
        norm = tf.reduce_sum(unnorm_new_weights, axis=0)

        w = tf.cond(tf.equal(norm, 0.),
                    true_fn=lambda: GMM['weights'],
                    false_fn=lambda: unnorm_new_weights/norm)
        if cutoff < 1.0:
            raise NotImplementedError
        else:
            new_weights = w
            means = GMM['means']
            covariances = GMM['covariances']

        K_11 = tf.gather_nd(covariances, mis_mis, name="K_11")
        K_12 = tf.gather_nd(covariances, mis_obs, name="K_12")
        K_21 = tf.gather_nd(covariances, obs_mis, name="K_21")
        K_1222 = K_12 @ K_22__1

        new_mean = (tf.gather_nd(means, mis)
                    + tf.squeeze(K_1222 @ right_diff, axis=2))
        new_covariance = K_11 - K_1222 @ K_21
        return [new_weights, new_mean, new_covariance, mis, obs, mis_mis,
                mis_obs, obs_mis, obs_obs]

    return tf.cond(tf.equal(n_missing, 0),
                   true_fn=lambda: [tf.ones([1], tf_float),
                                    tf.zeros([1, 0], tf_float),
                                    tf.zeros([1, 0, 0], tf_float),
                                    mis, obs, mis_mis, mis_obs,
                                    obs_mis, obs_obs],
                   false_fn=lambda: tf.cond(
                       tf.equal(not_n_missing, 0),
                       true_fn=lambda: [GMM['weights'], GMM['means'],
                                        GMM['covariances'], mis,
                                        obs, mis_mis,
                                        mis_obs, obs_mis, obs_obs],
                       false_fn=conditional_dist),
                   strict=True)


# Tested cursorily against numpy implementation
@add_variable_scope(name="tf_uncertain_point")
def tf_uncertain_point(inp, GMM, M):
    "M is already a diagonal matrix"
    mask = tf.is_nan(inp)
    weights, means, covariances, mis, obs, mis_mis, mis_obs, obs_mis, obs_obs \
        = tf_conditional_mog((inp, mask), GMM)
    n_gaussians = tf.shape(weights)[0]
    n_variables = int(GMM['means'].get_shape()[1])

    mu_obs = tf.gather(inp, obs[0, :, 1])[tf.newaxis, :, tf.newaxis]
    precisions = tf.matrix_inverse(covariances)
    means = means[:, :, tf.newaxis]
    sig_mu = precisions @ means
    mu_sig_mu = tf.matmul(means, sig_mu, transpose_a=True)

    A = precisions + tf.gather_nd(M, mis_mis[0, :, :, 1:])
    Ainv = tf.matrix_inverse(A)
    sqrt_det = tf.rsqrt(tf.matrix_determinant(covariances)
                        * tf.matrix_determinant(A))
    norm_factor = (weights * sqrt_det)[:, tf.newaxis, tf.newaxis]
    neg_Ainv_a = Ainv @ sig_mu
    a_Ainv_a = tf.matmul(sig_mu, neg_Ainv_a, transpose_a=True)

    obs_sc = tf.matmul(mu_obs, tf.gather_nd(M, obs_obs[:1, :, :, 1:]),
                       transpose_a=True) @ mu_obs
    sum_contrib = mu_sig_mu - a_Ainv_a + obs_sc
    M_mis = tf.gather_nd(M, mis[:n_gaussians, :, 1:])
    M_obs_one = tf.gather_nd(M, obs[:1, :, 1:])
    b = (tf.matmul(M_mis, neg_Ainv_a, transpose_a=True)
         + tf.matmul(M_obs_one, mu_obs, transpose_a=True))
    # M___mis = tp(tf.gather_nd(tp(M), M_mis_mask))
    __B = tf.matmul(M_mis, Ainv, transpose_a=True)
    B_2 = -(M - __B @ M_mis)

    # Now scatter the matrices back in order to be able to stack them
    s_sig_mu = tf.scatter_nd(
        indices=mis[:n_gaussians], updates=sig_mu, shape=tf.cast(
            tf.convert_to_tensor([n_gaussians, n_variables, 1]),
            dtype=tf.int64))
    s_mu_obs = tf.where(mask, x=tf.zeros_like(inp),
                        y=inp)[tf.newaxis, :, tf.newaxis]
    Cinv = tf.scatter_nd(indices=mis_mis[:n_gaussians], updates=Ainv,
                         shape=tf.cast(tf.convert_to_tensor(
                             [n_gaussians, n_variables, n_variables]),
                                       dtype=tf.int64))
    return [b, B_2, norm_factor, sum_contrib, mu_sig_mu, s_sig_mu, s_mu_obs,
            Cinv]


@add_variable_scope(name="tf_points_statistics")
def tf_points_statistics(inputs, GMM, M, parallel_iterations=50):
    tf_float = inputs.dtype
    num_elems = tf.shape(inputs)[0]
    stats = list(tf.TensorArray(tf_float, size=num_elems,
                                clear_after_read=False)
                 for _ in range(8))  # to accommodate all returned tensors

    def loop_body(i, *arrays):
        inp = inputs[i]
        point = tf_uncertain_point(inp, GMM, M)
        outs = map(lambda av: av[0].write(i, av[1]), zip(arrays, point))
        return (i + 1,) + tuple(outs)

    _, *arrs = tf.while_loop(
        lambda i, *_: i < num_elems, loop_body, [0]+stats)
    return arrs
    # return tf.map_fn(lambda i: tf_uncertain_point(i, GMM, M),
    #                  inputs, dtype=[inputs.dtype]*8,
    #                  parallel_iterations=parallel_iterations,
    #                  back_prop=True,
    #                  name="points_stats")


@add_variable_scope()
def k_f(b, B_2, x_norm, x_sum_contrib, v_mu_sig_mu, v_sig_mu, v_mu_obs, Cinv,
        v_norm, rbf_var):
    def make_X(t):
        assert len(t.get_shape()) == 3
        new_shape = [1] * 4
        new_shape[1] = tf.shape(v_sig_mu)[0]
        return tf.tile(t[:, tf.newaxis, ...], new_shape)

    def make_V(t):
        assert len(t.get_shape()) == 3
        new_shape = [1] * 4
        new_shape[0] = tf.shape(x_norm)[0]
        return tf.tile(t[tf.newaxis, ...], new_shape)

    b = make_X(b)
    B_2 = make_X(B_2)
    x_sum_contrib = make_X(x_sum_contrib)
    v_mu_sig_mu = make_V(v_mu_sig_mu)
    v_sig_mu = make_V(v_sig_mu)
    v_mu_obs = tf.tile(v_mu_obs[tf.newaxis, ...],
                       [tf.shape(x_norm)[0], tf.shape(v_sig_mu)[1], 1, 1])
    Cinv = make_V(Cinv)

    ww = v_norm[tf.newaxis, :] * x_norm[:, tf.newaxis]
    # remember `v_mu_obs` is scattered
    d = B_2 @ v_mu_obs
    c = tf.where(tf.equal(v_sig_mu, 0), x=v_sig_mu, y=(v_sig_mu + b + d))
    a = tf.matmul(c, Cinv, transpose_a=True) @ c
    e = tf.matmul(2*b + d, v_mu_obs, transpose_a=True)
    exp = x_sum_contrib + v_mu_sig_mu - a - e
    out = ww * tf.exp(-.5*exp)
    return tf.reduce_sum(out) * rbf_var


@add_variable_scope(name="kernel_fun")
def make_kernel_fun(input_dims, GMM, tf_float=tf.float64):
    tf_GMM = dict((l, tf.constant(value=v, dtype=tf_float))
                  for l, v in GMM.items())
    X = tf.placeholder(tf_float, [None, input_dims], name="X")
    lengthscale = tf.get_variable("lengthscale", shape=[input_dims],
                                  dtype=tf_float,
                                  initializer=tf.constant_initializer(1.0))
    rbf_var = tf.get_variable("rbf_var", shape=[], dtype=tf_float,
                              initializer=tf.constant_initializer(1.0))
    white_var = tf.get_variable("white_var", shape=[], dtype=tf_float,
                                initializer=tf.constant_initializer(1.0))
    M = tf.diag(lengthscale**(-2), name="M")

    b, B_2, norm, sum_contrib, v_mu_sig_mu, v_sig_mu, v_mu_obs, Cinv \
        = tf_points_statistics(X, tf_GMM, M)

    def strict_tril_indices(n):
        y, x = np.tril_indices(n)
        m = y != x
        return y[m].astype(np.int32), x[m].astype(np.int32)

    len_X = tf.shape(X)[0]
    tril_indices = tf.py_func(strict_tril_indices, [len_X],
                              (tf.int32, tf.int32), stateful=False,
                              name="tril_indices")

    K_symm_tril = tf.map_fn(
        lambda t: k_f(b.read(t[0]), B_2.read(t[0]), norm.read(t[0]),
                      sum_contrib.read(t[0]),
                      v_mu_sig_mu.read(t[1]), v_sig_mu.read(t[1]),
                      v_mu_obs.read(t[1]), Cinv.read(t[1]), norm.read(t[1]),
                      rbf_var, name="K_symm_f"),
        tril_indices, dtype=tf_float, parallel_iterations=10)

    K_symm = tf.scatter_nd(tf.stack(tril_indices, axis=1),
                           K_symm_tril, (len_X, len_X), "K_symm_scatter")
    K_symm = (K_symm + tf.transpose(K_symm)
              + tf.eye(len_X, dtype=tf_float) * (white_var + rbf_var))

    X2 = tf.placeholder(tf_float, [None, input_dims], name="X2")
    _, _, v_norm, _, v_mu_sig_mu, v_sig_mu, v_mu_obs, Cinv \
        = tf_points_statistics(X2, tf_GMM, M)
    len_X2 = tf.shape(X2)[0]

    matrix_indices = tf.reshape(tf.stack([
        tf.tile(tf.expand_dims(tf.range(len_X), 1), [1, len_X2]),
        tf.tile(tf.expand_dims(tf.range(len_X2), 0), [len_X, 1])],
                                         axis=2), (-1, 2))
    K_a_f = tf.map_fn(
        lambda t: k_f(b.read(t[0]), B_2.read(t[0]), norm.read(t[0]),
                      sum_contrib.read(t[0]),
                      v_mu_sig_mu.read(t[1]), v_sig_mu.read(t[1]),
                      v_mu_obs.read(t[1]), Cinv.read(t[1]), norm.read(t[1]),
                      rbf_var, name="K_a_f"),
        matrix_indices, dtype=tf_float, parallel_iterations=10)
    K_a = tf.reshape(K_a_f, [len_X, len_X2])

    dL_dK_ph = tf.placeholder(tf_float, [None, None], name="dL_dK_ph")
    l_g_symm, rbfv_g_symm = tf.gradients(
        tf.reduce_sum(dL_dK_ph * K_symm), [lengthscale, rbf_var],
        name='gradients_symm')
    l_g_a, rbfv_g_a = tf.gradients(
        tf.reduce_sum(dL_dK_ph * K_a), [lengthscale, rbf_var],
        name='gradients_a_syy')

    lengthscale_ph = tf.placeholder(tf_float, shape=[input_dims],
                                    name="lengthscale_ph")
    rbf_var_ph = tf.placeholder(tf_float, shape=[], name="rbf_var_ph")
    white_var_ph = tf.placeholder(tf_float, shape=[], name="white_var_ph")

    lengthscale_op = tf.assign(lengthscale, lengthscale_ph)
    rbf_var_op = tf.assign(rbf_var, rbf_var_ph)
    white_var_op = tf.assign(white_var, white_var_ph)
    assignment_ops = [lengthscale_op, rbf_var_op, white_var_op]

    return ((K_symm, l_g_symm, rbfv_g_symm), (K_a, l_g_a, rbfv_g_a),
            ((lengthscale_ph, rbf_var_ph, white_var_ph, dL_dK_ph, X, X2),
             assignment_ops))


class TFUncertainMoGRBFWhite(Kern):
    def __init__(self, input_dim, mog, active_dims=None,
                 name='tfUncertainMoG', **_kwargs):
        super(TFUncertainMoGRBFWhite, self).__init__(input_dim, active_dims,
                                                     name)
        self.input_dim = input_dim
        tf.reset_default_graph()
        symm, a_symm, assignments = make_kernel_fun(input_dim, mog)
        self.K_symm, self.l_g_symm, self.rbfv_g_symm = symm
        self.K_a, self.l_g_a, self.rbfv_g_a = a_symm

        (self.lengthscale_ph, self.rbf_var_ph, self.white_var_ph, self.dL_dK_ph,
         self.X_ph, self.X2_ph), self.assignment_ops = assignments

        self.white_var = Param('white_var', 1.0)
        self.rbf_var = Param('rbf_var', 1.0)
        self.lengthscale = Param('lengthscale',
                                 np.ones([input_dim], dtype=np.float64))
        self.link_parameters(self.white_var, self.rbf_var, self.lengthscale)

        self.sess = tf.Session()

    def parameters_changed(self):
        self.sess.run(self.assignment_ops, {
            self.lengthscale_ph: np.array(self.lengthscale),
            self.rbf_var_ph: np.array(self.rbf_var)[0],
            self.white_var_ph: np.array(self.white_var)[0],
        })

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None):
        if X2 is None:
            K = self.sess.run(self.K_symm, {self.X_ph: X})
        else:
            K = self.sess.run(self.K_a, {self.X_ph: X, self.X2_ph: X2})
        return K

    def Kdiag(self, X):
        return (self.rbf_var + self.white_var) * np.ones(len(X))

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None:
            self.lengthscale.gradient, self.rbf_var.gradient \
                = self.sess.run([self.l_g_symm, self.rbfv_g_symm], {
                    self.dL_dK_ph: dL_dK,
                    self.X_ph: X})
            self.white_var.gradient = np.trace(dL_dK)
        else:
            self.lengthscale.gradient, self.rbf_var.gradient \
                = self.sess.run([self.l_g_a, self.rbfv_g_a], {
                    self.dL_dK_ph: dL_dK,
                    self.X_ph: X,
                    self.X2_ph: X2})
            self.white_var.gradient = 0.0

    def update_gradients_diag(self, dL_dKdiag, X):
        self.lengthscale.gradient[...] = 0.0
        self.white_var.gradient = \
            self.rbf_var.gradient = np.sum(dL_dKdiag)

    def close(self):
        self.sess.close()


if __name__ == '__main__':
    import pickle_utils as pu

    X = np.random.rand(4, 13)
    mog = pu.load("impute_benchmark/imputed_BGMM_20_BostonHousing_MCAR_rows_0.7"
                  "/params.pkl.gz")
    mk = TFUncertainMoGRBFWhite(X.shape[1], mog)
    print(mk.K(X))
