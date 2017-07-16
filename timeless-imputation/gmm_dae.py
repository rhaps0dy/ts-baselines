"""
Deep Denoising Autoencoder, but the noise is introduced by a Gaussian Mixture
Model trained on the input data.
"""

import unittest
import numpy as np
import pandas as pd
import gmm_impute
import tensorflow as tf

from add_variable_scope import add_variable_scope

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor


@add_variable_scope()
def tf_mask_matrices(matrices, m1, m1_dim, m2, m2_dim):
    mask = tf.expand_dims(m1, -1) & m2

    shape = []
    if len(matrices.get_shape()) == 3:
        shape.append(int(matrices.get_shape()[0]))
        mask = tf.tile(mask, [tf.shape(matrices)[0], 1, 1])
    shape += [m1_dim, m2_dim]
    mat = tf.boolean_mask(matrices, mask)
    return tf.reshape(mat, shape)

@add_variable_scope(name="gaussian_pdf")
def tf_gaussian_pdf(diffs, right_diffs, covariances, inverse_covariances):
    """Compute the gaussian PDF for each row of `inputs`, and
    each element of GMM."""
    left_diffs = tf.expand_dims(diffs, axis=1)
    exponent = left_diffs @ inverse_covariances @ right_diffs

    exponent = -0.5 * tf.squeeze(exponent, axis=[1,2])
    dets = tf.matrix_determinant(covariances)
    divider = tf.pow(tf.constant(2 * np.pi, dtype=diffs.dtype),
                     tf.cast(tf.shape(diffs)[0], dtype=diffs.dtype))
    g_pdf = tf.exp(exponent) * tf.rsqrt(divider * dets)
    return g_pdf


@add_variable_scope(name="tf_gmm_impute")
def tf_gmm_impute(_inputs, GMM):
    "Imputes one sample of _inputs using the Gaussian Mixture Model `GMM`"
    inputs, mask = _inputs

    n_missing = tf.reduce_sum(tf.to_int32(mask), axis=0)
    not_n_missing = int(inputs.get_shape()[0]) - n_missing
    assert len(n_missing.get_shape()) == 0, "n_missing is a scalar"

    @add_variable_scope(name="conditional_dist")
    def conditional_dist():
        # Compute just enough Gaussian to compute `new_weights`
        not_mask = ~mask
        expanded_not_mask = tf.expand_dims(not_mask, axis=0)
        K_22 = tf_mask_matrices(GMM['covariances'],
                                expanded_not_mask, not_n_missing,
                                expanded_not_mask, not_n_missing,
                                name="K_22")
        K_22__1 = tf.matrix_inverse(K_22)
        known_inputs_diff = (tf.boolean_mask(inputs, not_mask) -
                             tf.transpose(tf.boolean_mask(
                                 tf.transpose(GMM['means']), not_mask)))
        right_diff = tf.expand_dims(known_inputs_diff, axis=2)
        g_pdf = tf_gaussian_pdf(known_inputs_diff, right_diff, K_22, K_22__1)
        new_weights = GMM['weights'] * g_pdf
        new_weights /= tf.reduce_sum(new_weights, axis=0)

        # Now draw a sample from the mixture categorical distribution
        dist_index = ds.Categorical(probs=new_weights).sample()

        # And compute the conditioned distribution from that Gaussian
        mean = GMM['means'][dist_index,:]
        covariance = GMM['covariances'][dist_index,:,:]
        K_11 = tf_mask_matrices(covariance,
                                mask, n_missing,
                                mask, n_missing,
                                name="K_11")
        K_12 = tf_mask_matrices(covariance,
                                mask, n_missing,
                                not_mask, not_n_missing,
                                name="K_12")
        K_21 = tf_mask_matrices(covariance,
                                not_mask, not_n_missing,
                                mask, n_missing,
                                name="K_21")
        K_1222 = K_12 @ K_22__1[dist_index,:,:]

        diff = right_diff[dist_index,:,:]
        new_mean = tf.boolean_mask(mean, mask) + tf.squeeze(K_1222 @ diff, axis=-1)
        new_covariance = K_11 - K_1222 @ K_21

        # cholesky() ignores the upper-triangular part, so we can use it to
        # avoid problems caused by loss of precision.
        # Sometimes matrices are not symmetric because of the ~15th digit
        L_covariance = tf.cholesky(new_covariance)
        imputing_gaussian = ds.MultivariateNormalTriL(
            loc=new_mean,
            scale_tril=L_covariance,
            validate_args=True,
            allow_nan_stats=False,
            name="imputing_gaussian")

        # Use the conditioned distribution to impute the inputs
        scattered = tf.scatter_nd(tf.where(mask),
                                  imputing_gaussian.sample(),
                                  mask.get_shape())
        return tf.where(mask, scattered, inputs)

    return tf.cond(tf.equal(n_missing, 0),
                   true_fn=lambda: inputs,
                   false_fn=conditional_dist,
                   strict=True)

#@add_variable_scope(name="autoencoder")
def autoencoder(inputs, targets):
    return inputs


@add_variable_scope(name="gmm_dae_model")
def gmm_dae_model(inputs, GMM, inputs_drop_prob):
    random_mask = st.StochasticTensor(
        ds.Bernoulli(probs=inputs_drop_prob, dtype=tf.bool),
        dist_value_type=st.SampleValue(tf.shape(inputs)),
        name="random_mask")
    mask = random_mask | tf.is_nan(inputs)
    tf_GMM = dict((l, tf.constant(value=v, dtype=inputs.dtype))
                  for l, v in GMM.items())

    preimputed_inputs = tf.map_fn(lambda i: tf_gmm_impute(i, tf_GMM),
                                  [inputs, mask],
                                  dtype=inputs.dtype,
                                  parallel_iterations=10,
                                  back_prop=False,
                                  name="preimputed_inputs")

    outputs, loss = autoencoder(preimputed_inputs, inputs)
    return outputs


def gmm_dae_impute(log_dir, dataset, GMM, n_impute=100):
    # # Code to simultaneously normalise dataset and GMM (now unused)
    # inputs_mean = dataset.mean().values
    # inputs_std = dataset.std().values
    # inputs = (dataset.values - inputs_mean) / inputs_std
    #
    # normalised_means = (GMM['means'] - inputs_mean) / inputs_std
    # # Turn normalisation into affine transformation
    # A = np.eye(len(inputs_std), len(inputs_std)) / inputs_std
    #    normalised_covariances = A @ GMM['covariances'] @ A.T

    # ndarray = gmm_impute._gmm_impute({'means': normalised_means,
    # 'covariances': normalised_covariances, 'weights': GMM['weights']}, inputs)

    imp = gmm_dae_model(tf.constant(dataset.values, dtype=tf.float64), GMM, 0.)
    ndarray = []
    with tf.Session() as sess:
        for _ in range(n_impute):
            ndarray.append(sess.run(imp))
    return list(pd.DataFrame(a,  # *inputs_std + inputs_mean,
                             index=dataset.index,
                             columns=dataset.columns) for a in ndarray)

class TestGMMImpute(unittest.TestCase):
    def test_full_impute_equal(self):
        d = {'means': tf.constant(np.array([np.arange(3, dtype=np.float)] * 5)),
             'covariances': tf.constant(np.array([[np.arange(3, dtype=np.float)] * 3] * 5)),
             'weights': tf.constant(np.array([1 / 5] * 5, dtype=np.float))}
        self.assertEqual(tuple(d['means'].shape), (5, 3))
        self.assertEqual(tuple(d['covariances'].shape), (5, 3, 3))
        self.assertEqual(tuple(d['weights'].shape), (5,))
        ph = tf.placeholder(d['means'].dtype, shape=[3])
        imp = tf_gmm_impute((ph, tf.constant([False,False,False])), d)

        with tf.Session() as sess:
            for _ in range(10):
                a = np.random.rand(3)
                self.assertTrue(np.all(a == sess.run(imp, {ph: a})))
