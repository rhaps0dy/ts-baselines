import datasets
import pickle_utils as pu
import utils
import numpy as np
import unittest
import tensorflow as tf
from tensorflow.python.framework import tensor_shape, tensor_util
import os
from tqdm import tqdm
import time
import itertools
import category_dae
from add_variable_scope import add_variable_scope
import pandas as pd
import selu


def MSRA_initializer(dtype=tf.float64):
    return tf.contrib.layers.variance_scaling_initializer(factor=2.0)


@add_variable_scope(name="FC_net")
def FC_net(inputs, layer_sizes_nlin, init, keep_prob, dropout_last_layer=False,
           collection=None):
    """Constructs a fully-connected network to specification.
    `layer_sizes_nlin` includes the size of each hidden layer and its
    activation function. It does not include the size of the input, which is
    deduced from `inputs`."""
    tf_float = inputs.dtype

    ls_out, nonlinearities = zip(*layer_sizes_nlin)
    ls_in = [int(inputs.get_shape()[1])] + list(ls_out[:-1])
    training = tf.equal(keep_prob, 1., name="training")
    #x = tf.check_numerics(inputs, "inputs")
    x = inputs
    for i, (m, n, nlin) in enumerate(zip(ls_in, ls_out, nonlinearities)):
        with tf.variable_scope("layer_{:d}".format(i)):
            #if nlin == selu.nlin and i > 0:
            #    x = selu.dropout(x, keep_prob, name="dropout_selu_{:d}".format(i),
            #                     training=training)
            #else:
            #    x = tf.nn.dropout(x, keep_prob, name="dropout_{:d}".format(i))

            W = tf.get_variable("W", shape=[m, n],
                                initializer=init, dtype=tf_float)
            b = tf.get_variable("b", shape=[n],
                                initializer=tf.constant_initializer(
                                    0.0 if nlin != tf.nn.relu else 0.1),
                                dtype=tf_float)
            if collection is not None:
                tf.add_to_collection(collection, W)
                tf.add_to_collection(collection, b)
            #x = tf.check_numerics(x, "before_linout_{:d}".format(i))
            #W = tf.check_numerics(W, "W_{:d}".format(i))
            #b = tf.check_numerics(b, "b_{:d}".format(i))
            x = tf.matmul(x, W) + b
            #x = tf.check_numerics(x, "linout_{:d}".format(i))
            if nlin is not None:
                x = nlin(x, name="activation_{:d}".format(i))
            #x = tf.check_numerics(x, "layer_{:d}".format(i))
    #if dropout_last_layer:
    #    if nlin == selu.nlin:
    #        x = selu.dropout(x, keep_prob, name="dropout_selu_{:d}".format(i),
    #                         training=training)
    #    else:
    #        x = tf.nn.dropout(x, keep_prob, name="dropout_{:d}".format(i))
    return x


###
# FUNCTIONS SPECIFIC TO IMPUTATION
###


def preimpute_without_model(rs, batch, possible_inputs, possible_inputs_lens,
                            additional_corruption_p=0.0):
    if possible_inputs is None:
        return [batch, batch == 0]
    batch = batch.copy()
    if batch.dtype == np.float64:
        nan_mask = np.isnan(batch)
    else:
        nan_mask = batch == category_dae.post_NA_int32
    if additional_corruption_p > 0.0:
        nan_mask |= rs.random_sample(batch.shape) < additional_corruption_p
    where_nans = np.nonzero(nan_mask)
    # Draw a uniformly random index for all data points
    chosen_indices = np.floor(rs.random_sample(where_nans[1].shape) *
                              possible_inputs_lens[where_nans[1]]).astype(
                                  np.int_)
    batch[nan_mask] = possible_inputs[where_nans[1], chosen_indices]
    return batch, nan_mask


@add_variable_scope(name="tf_network")
def tf_network(layer_sizes, input_layer, pristine_num, pristine_cat, init,
               nlin, residual, keep_prob):
    tf_float = input_layer["inputs"].dtype
    inputs = tf.concat([input_layer["inputs"], tf.cast(
        input_layer["mask_inputs"], tf_float)], axis=1)

    # Append output layer size
    num_size = len(input_layer["num_idx"])
    layer_sizes.append(num_size + sum(input_layer["n_cats_l"]))
    nonlinearities = [nlin] * (len(layer_sizes)-1) + [None]

    # Create fully-connected network part
    outputs = FC_net(inputs, list(zip(layer_sizes, nonlinearities)),
                     init, keep_prob)

    # Concatenate outputs and define loss
    ret = {}
    cat_preds = []
    cat_losses = []
    prev_size = num_size
    for i, n_cats in enumerate(input_layer["n_cats_l"]):
        this_slice = outputs[:, prev_size:prev_size+n_cats]
        cat_preds.append(tf.argmax(this_slice, axis=1) + 1)
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=input_layer["cat_inputs"][:, i],
            logits=this_slice)
        assert len(l.get_shape()) == 1
        cat_losses.append(l)
        prev_size += n_cats
    assert prev_size == int(outputs.get_shape()[1]), "we used all the outputs"

    if num_size == 0:
        loss = None
    else:
        num_preds = outputs[:, :num_size]
        num_mask = ~tf.is_nan(pristine_num)
        diff = tf.squared_difference(tf.boolean_mask(pristine_num, num_mask),
                                     tf.boolean_mask(num_preds, num_mask))
        loss = tf.reduce_mean(diff)
        ret["num_preds"] = num_preds

    if len(input_layer["cat_idx"]) > 0:
        cat_preds = tf.stack(cat_preds, axis=1)
        cat_loss = tf.boolean_mask(tf.stack(cat_losses, axis=1),
                                   tf.not_equal(pristine_cat,
                                                category_dae.post_NA_int32),
                                   name="cat_loss")
        if loss is None:
            loss = tf.reduce_mean(cat_loss)
        else:
            loss += tf.reduce_mean(cat_loss)
        ret["cat_preds"] = cat_preds

    ret["loss"] = loss
    return ret


def collect_possible_inputs(df, dtype):
    if df.shape[1] == 0:
        return None, None

    possible_inputs_l = []
    for i in range(df.shape[1]):
        v = df.iloc[:, i].values
        assert v.dtype == dtype
        if v.dtype == np.float64:
            mask = ~np.isnan(v)
        elif v.dtype == np.int32:
            mask = (v != utils.NA_int32)
        else:
            raise ValueError(v.dtype.name)
        possible_inputs_l.append(v[mask])

    possible_inputs_lens = np.array(list(map(len, possible_inputs_l)),
                                    dtype=np.float64)
    possible_inputs = np.zeros(shape=[len(possible_inputs_l),
                                      int(np.max(possible_inputs_lens))],
                               dtype=dtype)
    for i, l in enumerate(possible_inputs_l):
        possible_inputs[i, :len(l)] = l
    del possible_inputs_l
    return possible_inputs, possible_inputs_lens


def impute(log_path, dataset, full_data, number_imputations=5,
           number_layers=16, hidden_units=512, seed=0,
           init_type=MSRA_initializer, batch_size=256,
           nlin=tf.nn.relu, residual=False,
           patience=50, learning_rate=0.001, optimizer=tf.train.AdamOptimizer,
           corruption_prob=0.5, keep_prob=0.95, validation_proportion=0.15,
           tf_float=tf.float64):
    d = {}
    for k, v in locals().items():
        if k not in {'log_path', 'dataset', 'full_data'}:
            d[k] = str(v)
    with open(os.path.join(log_path, 'flags.txt'), 'w') as f:
            f.write(str(d)+'\n')
    del d, k, v, f

    tf.reset_default_graph()
    rs = np.random.RandomState(seed)
    tf.set_random_seed(rs.randint(999999))

    input_layer = category_dae.make_input_layer(dataset, tf_float=tf_float)
    test_df = category_dae.preprocess_dataframe(dataset[0], input_layer)

    validation_mask = rs.rand(len(test_df)) < validation_proportion
    train_df = test_df[~validation_mask]
    validation_df = test_df[validation_mask]

    num_batches = int(np.ceil(train_df.shape[0] / batch_size))

    pristine_num = tf.placeholder(tf_float,
                                  shape=[None, len(input_layer["num_idx"])],
                                  name="pristine_num")
    pristine_cat = tf.placeholder(tf.int32,
                                  shape=[None, len(input_layer["cat_idx"])],
                                  name="pristine_cat")
    keep_prob_ph = tf.placeholder(tf_float, shape=[], name="keep_prob")
    layer_sizes = [hidden_units] * (number_layers - 1)
    loss_preds = tf_network(layer_sizes, input_layer, pristine_num,
                            pristine_cat, init_type, nlin, residual,
                            keep_prob_ph)

    # Make the input data distribution
    num_pi = collect_possible_inputs(test_df[input_layer["num_idx"]],
                                     np.float64)
    if num_pi[0] is None:
        print(log_path, ": num_pi is None")
    cat_pi = collect_possible_inputs(test_df[input_layer["cat_idx"]],
                                     np.int32)
    if cat_pi[0] is None:
        print(log_path, ": cat_pi is None")

    loss = loss_preds["loss"]
    train_num = train_df[input_layer["num_idx"]].values
    train_cat = train_df[input_layer["cat_idx"]].values
    validation_num = validation_df[input_layer["num_idx"]].values
    validation_cat = validation_df[input_layer["cat_idx"]].values

    test_num = test_df[input_layer["num_idx"]].values
    test_full_num = full_data[0][input_layer["num_idx"]].values
    test_num_norm_min = test_full_num.min(axis=0)
    test_num_norm_max = (test_full_num - test_num_norm_min).max(axis=0)
    test_full_num_norm = ((test_full_num - test_num_norm_min) /
                          test_num_norm_max)
    test_cat = test_df[input_layer["cat_idx"]].values

    train_op = optimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=10)
    training_summary = tf.summary.scalar("loss/training", loss)
    validation_summary = tf.summary.scalar("loss/validation", loss)
    rmse_ph = tf.placeholder(tf.float64, shape=[])
    rmse_summary = tf.summary.scalar("test/RMSE", rmse_ph)
    pfc_ph = tf.placeholder(tf.float64, shape=[])
    pfc_summary = tf.summary.scalar("test/PFC", pfc_ph)
    test_summary = tf.summary.merge([rmse_summary, pfc_summary])

    def compute_impute(sess, test_num=test_num, test_cat=test_cat):
        net_outs = []
        for imp_i in range(16):
            imp_d_num, imp_d_num_mask = preimpute_without_model(
                rs, test_num, *num_pi)
            imp_d_cat, imp_d_cat_mask = preimpute_without_model(
                    rs, test_cat, *cat_pi)
            for it in range(number_imputations // 16):
                feed_dict = {keep_prob_ph: 1.0}
                to_run = [[], []]
                if "num_inputs" in input_layer:
                    feed_dict[input_layer["num_inputs"]] = imp_d_num
                    feed_dict[input_layer["num_mask_inputs"]] = imp_d_num_mask
                    to_run[0] = loss_preds["num_preds"]
                if "cat_inputs" in input_layer:
                    feed_dict[input_layer["cat_inputs"]] = imp_d_cat
                    feed_dict[input_layer["cat_mask_inputs"]] = imp_d_cat_mask
                    to_run[1] = loss_preds["cat_preds"]
                imp_d_num, imp_d_cat = sess.run(to_run, feed_dict)

            net_outs.append(pd.concat([
                pd.DataFrame(imp_d_num, columns=input_layer["num_idx"]),
                pd.DataFrame(imp_d_cat, columns=input_layer["cat_idx"])
            ], axis=1))
        return net_outs, imp_d_num_mask

    def into_feed_dict(key, original, pi, pristine, fd):
        ka = key + "_inputs"
        kb = key + "_mask_inputs"
        if ka in input_layer:
            corrupted, mask = preimpute_without_model(
                rs, original, *pi,
                additional_corruption_p=corruption_prob)
            fd[input_layer[ka]] = corrupted
            fd[input_layer[kb]] = mask
            fd[pristine] = original

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(log_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        time_since_last_save = time.time()
        time_since_last_log = time.time()
        last_validation_loss = np.inf
        remaining_patience = patience
        just_restored = False

        ckpt = tf.train.latest_checkpoint(log_path)
        if ckpt is not None:
            print("Restoring model from", ckpt)
            saver.restore(sess, ckpt)
            #remaining_patience = 0
            just_restored = True
        del ckpt

        for step in tqdm(itertools.count(0)):
            if remaining_patience <= 0:
                if not just_restored:
                    print("Saving checkpoint...")
                    #saver.save(sess, os.path.join(log_path, 'ckpt'),
                    #           global_step=step)
                break

            i = step % num_batches
            # Populate feed_dict; duplicated code for num and cat
            feed_dict = {keep_prob_ph: keep_prob}
            into_feed_dict("num",
                           train_num[batch_size * i:batch_size * (i + 1)],
                           num_pi, pristine_num, feed_dict)
            into_feed_dict("cat",
                           train_cat[batch_size * i:batch_size * (i + 1)],
                           cat_pi, pristine_cat, feed_dict)
            del i

            #current_time = time.time()
            if step % 500 == 0:
                # Show loss and validation loss
                _, l, l_s = sess.run([train_op, loss, training_summary],
                                     feed_dict)
                summary_writer.add_summary(l_s, step)
                val_fd = {keep_prob_ph: 1.0}
                into_feed_dict("num", validation_num, num_pi, pristine_num,
                               val_fd)
                into_feed_dict("cat", validation_cat, cat_pi, pristine_cat,
                               val_fd)
                v_l, v_l_s = sess.run([loss, validation_summary], val_fd)
                if v_l < last_validation_loss:
                    remaining_patience = patience
                    last_validation_loss = v_l
                else:
                    remaining_patience -= 1
                summary_writer.add_summary(v_l_s, step)
                print("Loss: {}, validation loss: {}".format(l, v_l))

                # Compute test performance every 4 minutes
                #if False and current_time - time_since_last_save > 240.:
                if step % 10000 == 10000-500:
                    time_since_last_save = time.time()
                    print("Computing imputation performance...")
                    net_outs, imp_d_num_mask = compute_impute(sess)
                    if len(input_layer["num_idx"]) > 0:
                        num_outs = list(df[input_layer["num_idx"]].values
                                        for df in net_outs)
                        rmse = utils.mean_rmse(
                            imp_d_num_mask, test_full_num_norm,
                            (num_outs - test_num_norm_min) / test_num_norm_max)
                        print("The RMSE is:", rmse)
                    else:
                        rmse = np.nan
                    if len(input_layer["cat_idx"]) > 0:
                        pfc = datasets.percentage_falsely_classified(
                            test_df, full_data[0], net_outs)
                        print("The PFC is:", pfc)
                    else:
                        pfc = (np.nan, np.nan)
                    s = sess.run(test_summary, {rmse_ph: rmse,
                                                pfc_ph: pfc[0]/pfc[1]})
                    summary_writer.add_summary(s, step)

                time_since_last_log = time.time()

            else:
                _ = sess.run(train_op, feed_dict)

        imputed_dfs = compute_impute(sess)[0]
        return list(map(
            lambda df: category_dae.postprocess_dataframe(df, input_layer),
            imputed_dfs))


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
    dset, cat_idx = datasets.datasets()["LetterRecognition"]
    missing_dset = pu.load("impute_benchmark/"
                           "amputed_LetterRecognition_MCAR_total_0.3.pkl.gz")
    # missing_dset = utils.mcar_total(dset, missing_proportion=0.2)
    (missing_dset, dset), _ = utils.normalise_dataframes(missing_dset, dset,
                                                         method='mean_std')

    for optimizer in [tf.train.GradientDescentOptimizer]:
        for number_layers in [8]:
            for hidden_units in [128]:  # [128, 256, 512]:
                for corruption_prob in [.2]:  # [.2, .5, .8]:
                    for nlin in [selu.nlin]:  # [selu, tf.nn.relu, tf.nn.tanh]:
                        for keep_prob in [.95]:
                            dir_name = "DAE_selu_logs/{}_{}_{}_{}_{}_{}".format(
                                optimizer.__name__, number_layers, hidden_units,
                                corruption_prob, nlin.__name__, keep_prob)

                            if nlin == tf.nn.relu:
                                init_type = MSRA_initializer
                            else:
                                init_type = selu.initializer

                            impute(dir_name,
                                   (missing_dset, cat_idx),
                                   (dset, cat_idx),
                                   number_imputations=128,
                                   number_layers=number_layers,
                                   hidden_units=hidden_units,
                                   seed=10,
                                   init_type=init_type,
                                   batch_size=256,
                                   nlin=nlin,
                                   residual=False,
                                   patience=20,
                                   learning_rate=1e-3,
                                   keep_prob=keep_prob,
                                   corruption_prob=corruption_prob,
                                   optimizer=optimizer)
