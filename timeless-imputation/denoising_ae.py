import datasets
import pickle_utils as pu
import utils
import numpy as np
import unittest
import tensorflow as tf
import os
import time
import itertools


def MSRA_initializer(dtype=tf.float64):
    return tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                          dtype=dtype)


def SELU_initializer(dtype=tf.float64):
    return tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                          dtype=dtype)


def preimpute_without_model(rs, batch, possible_inputs, possible_inputs_lens,
                            additional_corruption_p=0.0):
    batch = batch.copy()
    nan_mask = np.isnan(batch)
    if additional_corruption_p > 0.0:
        nan_mask |= rs.random_sample(batch.shape) < additional_corruption_p
    where_nans = np.nonzero(nan_mask)
    # Draw a uniformly random index for all data points
    chosen_indices = np.floor(rs.random_sample(where_nans[1].shape) *
                              possible_inputs_lens[where_nans[1]]).astype(
                                  np.int_)
    batch[nan_mask] = possible_inputs[where_nans[1], chosen_indices]
    return batch, nan_mask

#def preimpute_without_model_tensorflow ( OBSOLETE )
#    possible_inputs = tf.constant(possible_inputs, dtype=tf.float64)
#    possible_inputs_lens = tf.constant(possible_inputs_lens, dtype=tf.float64)
#    nan_mask = tf.is_nan(inputs)
#    where_nans = tf.where(nan_mask)
#    feature_indices = tf.to_int32(where_nans[:,1])
#    print(possible_inputs_lens.get_shape(), feature_indices.get_shape())
#    chosen_indices = tf.to_int32(tf.random_uniform([tf.shape(where_nans)[0]],
#    dtype=tf.float64) *
#    tf.gather_nd(possible_inputs_lens, tf.expand_dims(feature_indices,
#    axis=1)))
#    replacements = tf.scatter_nd(where_nans, tf.gather_nd(possible_inputs,
#    tf.transpose([feature_indices, chosen_indices])),
#    tf.to_int64(tf.shape(inputs)))
#    return tf.where(nan_mask, replacements, inputs)

def tf_network(layer_sizes, original_inputs, corrupted_inputs, init, nlin, residual):
    x = corrupted_inputs
    for i, (m, n) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        with tf.variable_scope("layer_{:d}".format(i)):
            x = tf.check_numerics(x, "layer_{:d}".format(i))
            if i > 0:
                x = nlin(x)
            W = tf.get_variable("W", shape=[m, n],
                                initializer=init(tf.float64), dtype=tf.float64)
            b = tf.get_variable("b", shape=[n],
                                initializer=tf.constant_initializer(0.1),
                                dtype=tf.float64)
            if residual and m == n:
                x = x + tf.matmul(x, W) + b
            else:
                x = tf.matmul(x, W) + b
            x = tf.check_numerics(x, "layer_{:d}".format(i))

    with tf.variable_scope("loss"):
        mask = ~tf.is_nan(original_inputs)
        diff = tf.squared_difference(tf.boolean_mask(original_inputs, mask),
                                     # x is `corrupted_inputs` reconstructed
                                     tf.boolean_mask(x, mask))
        with tf.control_dependencies([tf.assert_greater(tf.shape(diff)[0], 0, [tf.shape(diff)])]):
            diff = tf.check_numerics(diff, "diff")
        return tf.reduce_mean(diff), x

def selu(x, name="selu"):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    with tf.variable_scope(name):
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def impute(log_path, dataset, full_dataset, number_imputations=5,
           number_layers=16, hidden_units=512, seed=0,
           init_type=MSRA_initializer, batch_size=256, num_epochs=10000000000000,
           nlin=tf.nn.relu, residual=False, patience=100, learning_rate=0.001,
           optimizer=tf.train.AdamOptimizer, corruption_prob=0.5):
    rs = np.random.RandomState(seed)
    tf.set_random_seed(rs.randint(999999))
    validation_mask = rs.rand(len(dataset)) < 0.15
    inputs = dataset.values[~validation_mask]
    validation_inputs = dataset.values[validation_mask]

    test_inputs = dataset.values
    test_original_inputs = full_dataset.values

    layer_sizes = [inputs.shape[1]] + ([hidden_units] * (
        number_layers - 1)) + [inputs.shape[1]]
    num_batches = int(np.ceil(inputs.shape[0] / batch_size))

    possible_inputs_l = [inputs[~np.isnan(inputs[:,i]),i]
                         for i in range(inputs.shape[1])]
    possible_inputs_lens = np.array(list(map(len, possible_inputs_l)),
                                    dtype=np.float64)
    possible_inputs = np.zeros(shape=[len(possible_inputs_l),
                                      int(np.max(possible_inputs_lens))],
                               dtype=np.float64)
    for i, l in enumerate(possible_inputs_l):
        possible_inputs[i, :len(l)] = l
    del possible_inputs_l

    original_batch_ph = tf.placeholder(tf.float64,
                                       shape=[None, inputs.shape[1]])
    corrupted_batch_ph = tf.placeholder(tf.float64,
                                        shape=[None, inputs.shape[1]])
    loss, preds = tf_network(layer_sizes, original_batch_ph, corrupted_batch_ph,
                             init_type, nlin, residual)
    train_op = optimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=0)
    training_summary = tf.summary.scalar("loss/training", loss)
    validation_summary = tf.summary.scalar("loss/validation", loss)
    rmse_ph = tf.placeholder(tf.float64, shape=[])
    rmse_summary = tf.summary.scalar("rmse/test", rmse_ph)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        time_since_last_save = time.time()
        time_since_last_log = time.time()
        last_validation_loss = np.inf
        remaining_patience = patience
        saver.restore(sess, 'DAE_logs/2_256_False_tanh/ckpt-291612')

        for step in itertools.count(0):
            #if remaining_patience <= 0:
            #    break
            i = step % num_batches
            original_batch = inputs[batch_size * i:batch_size * (i + 1)]
            corrupted_batch, _ = preimpute_without_model(
                rs, original_batch, possible_inputs, possible_inputs_lens,
                additional_corruption_p=corruption_prob)
            del i
            feed_dict = {original_batch_ph: original_batch,
                         corrupted_batch_ph: corrupted_batch}

            current_time = time.time()
            if current_time - time_since_last_log > 2.:
                # Show loss and validation loss
                _, l, l_s = sess.run([train_op, loss, training_summary],
                                     feed_dict)
                corrupted_validation, _ = preimpute_without_model(
                    rs, validation_inputs, possible_inputs,
                    possible_inputs_lens, additional_corruption_p=corruption_prob)
                summary_writer.add_summary(l_s, step)
                v_l, v_l_s = sess.run([loss, validation_summary],
                                      {original_batch_ph: validation_inputs,
                                       corrupted_batch_ph: corrupted_validation})
                if v_l < last_validation_loss:
                    remaining_patience = patience
                    last_validation_loss = v_l
                else:
                    remaining_patience -= 1
                summary_writer.add_summary(v_l_s, step)
                print("Loss: {}, validation loss: {}".format(l, v_l))

                # Save every 2 minutes
                if True or current_time - time_since_last_save > 120.:
                    #print("Saving checkpoint...")
                    #saver.save(sess, os.path.join(log_path, 'ckpt'),
                    #           global_step=step)
                    #time_since_last_save = time.time()
                    print("Computing imputation performance...")
                    net_outs = []
                    for imp_i in range(number_imputations):
                        imp_d, dwn = preimpute_without_model(
                            rs, test_inputs, possible_inputs,
                            possible_inputs_lens)
                        net_outs.append(sess.run(preds, {corrupted_batch_ph: imp_d}))
                    rmse = utils.mean_rmse(dwn, test_original_inputs, net_outs)
                    print("The RMSE is:", rmse)
                    s = sess.run(rmse_summary, {rmse_ph: rmse})
                    summary_writer.add_summary(s, step)
                time_since_last_log = time.time()

            else:
                sess.run(train_op, feed_dict)

    #imp_inputs = []
    #for _ in range(number_imputations):
    #    imp_d, dwn = preimpute_without_model(rs, dataset.values, possible_inputs,
    #                                         possible_inputs_lens)
    #    net_out = sess.run(preds, {original_batch_ph: imp_d, corrupted_batch_ph: imp_d})
    #    net_out[~dwn] = dataset.values[~dwn]
    #    imp_inputs.append(dataset.dataframe_like(dataset, net_out))
    #return imp_inputs


class TestDenoisingAE(unittest.TestCase):
    def test_preimpute_without_model(self):
        batch = np.arange(5 * 3).reshape([5, 3]).astype(np.float64)
        batch[[0, 2, 4], [2, 0, 1]] = np.nan
        possible_inputs = np.array([[0, 1, 2], [3, 4, -1], [5, -1, -1]]) + 0.5
        possible_inputs_lens = np.array([3, 2, 1])

        rs = np.random.RandomState(0)
        args = [rs, batch, possible_inputs, possible_inputs_lens]
        imputed_batch_1 = preimpute_without_model(*args)
        imputed_batch_2 = preimpute_without_model(*args)

        self.assertFalse(np.all(imputed_batch_1 == imputed_batch_2))
        self.assertFalse(np.any(np.isnan(imputed_batch_1)))
        self.assertFalse(np.any(np.isnan(imputed_batch_2)))
        self.assertTrue(np.all(imputed_batch_1 >= 0))
        self.assertTrue(np.all(imputed_batch_2 >= 0))
        self.assertTrue(np.all(imputed_batch_1[[0, 2, 4], [2, 0, 1]] % 1 == .5))
        self.assertTrue(np.all(imputed_batch_2[[0, 2, 4], [2, 0, 1]] % 1 == .5))


if __name__ == '__main__':
    dset = datasets.datasets()["BostonHousing"]
    missing_dset = pu.load("impute_benchmark/"
                           "amputed_BostonHousing_MCAR_total_0.3.pkl.gz")
    # missing_dset = utils.mcar_total(dset, missing_proportion=0.2)
    missing_dset, dset = utils.normalise_dataframes(missing_dset, dset)

    for number_layers in [2]:  # [2, 8, 16]:
        for hidden_units in [256]:  # [128, 256, 512]:
            for residual in [False]:  # [True, False]:
                for nlin in [tf.nn.tanh]:  # [selu, tf.nn.relu, tf.nn.tanh]:
                    dir_name = "DAE_logs/{:d}_{:d}_{}_{:s}".format(
                        number_layers, hidden_units, residual, nlin.__name__)
                    if not os.path.exists(dir_name):
                        os.mkdir(dir_name)
                    if nlin == tf.nn.relu:
                        init_type = MSRA_initializer
                    else:
                        init_type = SELU_initializer

                    tf.reset_default_graph()
                    impute(dir_name, missing_dset, dset,
                           number_imputations=128,
                           number_layers=number_layers,
                           hidden_units=hidden_units,
                           seed=0,
                           init_type=init_type,
                           batch_size=256,
                           nlin=nlin,
                           residual=residual,
                           patience=100000,
                           learning_rate=1e-4,
                           corruption_prob=0.2,
                           optimizer=tf.train.GradientDescentOptimizer)
