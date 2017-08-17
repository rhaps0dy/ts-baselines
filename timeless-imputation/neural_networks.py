import tensorflow as tf
import pickle_utils as pu
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('log_dir', '', 'Directory to log to')
    flags.DEFINE_integer('batch_size', 64, 'batch size for training')
    flags.DEFINE_integer('num_epochs', 1000, 'number of training epochs')
    flags.DEFINE_integer('increment', 7, 'number of training epochs')
    flags.DEFINE_integer('num_layers', 6, 'number of training epochs')
    flags.DEFINE_integer('patience', 20, 'number of training epochs')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_float('input_dropout', 0.5, 'droppo')
    del flags
FLAGS = tf.app.flags.FLAGS


def layer(inputs, num_units, name, nlin=None, trainable=True):
    print(name, num_units, "hidden units")
    with tf.variable_scope(name):
        W = tf.get_variable("W", [inputs.get_shape()[1], num_units],
                            dtype=tf.float32,
                            trainable=trainable,
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [num_units],
                            dtype=tf.float32,
                            trainable=trainable,
                            initializer=tf.constant_initializer(1.))
        x = tf.nn.bias_add(tf.matmul(inputs, W), b, name="activation")
        if nlin is not None:
            x = nlin(x)
        return x

def tf_rmse_sum_test(mask_missing, original, imputed):
    "Tensorflow RMSE_sum as in Multiple Imputation Using Deep Denoising Autoencoders (Gondara & Wang 2017)"
    assert mask_missing.get_shape() == original.get_shape()
    assert imputed.get_shape() == original.get_shape()
    with tf.variable_scope("rmse_sum"):
        sq_diff = tf.squared_difference(original, imputed)
        to_sum = tf.where(mask_missing, sq_diff, tf.zeros_like(sq_diff))
        per_attribute_rmse = tf.reduce_sum(sq_diff, axis=0)**.5
        return tf.reduce_sum(per_attribute_rmse)

def mse_sum_train(mask_missing, original, imputed):
    with tf.variable_scope("loss"):
        sq_diff = tf.squared_difference(original, imputed)
        return tf.reduce_mean(tf.reduce_sum(sq_diff, axis=1), axis=0)**.5

def autoencoder_impute(inputs_ph, mask_missing_ph, nlin=tf.nn.tanh, impute_input=True, trainable=True, name="gondara_impute", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        n_features = int(inputs_ph.get_shape()[1])
        if impute_input:
            # Remember that the dataset has zero mean.
            dataset_mean = tf.zeros_like(inputs_ph)
            x = tf.where(mask_missing_ph, dataset_mean, inputs_ph, name="ampute")
        else:
            x = inputs_ph
        if trainable:
            x = tf.nn.dropout(x, keep_prob=FLAGS.input_dropout)
        for i in range(1, FLAGS.num_layers+1):
            x = layer(x, n_features + i*FLAGS.increment, "layer_{:d}".format(i), nlin=nlin, trainable=trainable)
            x = tf.nn.dropout(x, keep_prob=FLAGS.input_dropout)
        for i in range(FLAGS.num_layers+1, 2*FLAGS.num_layers):
            x = layer(x, n_features + (2*FLAGS.num_layers-i)*FLAGS.increment, "layer_{:d}".format(i), nlin=nlin, trainable=trainable)
            x = tf.nn.dropout(x, keep_prob=FLAGS.input_dropout)
        x = layer(x, n_features, "layer_out", trainable=trainable)
        autoencoder_preds = x
        return autoencoder_preds, name

def autoencoder_dropout(inputs, mask_missing, nlin=tf.nn.tanh, impute_input=True, trainable=True, name="dropout_impute", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        n_features = int(inputs_ph.get_shape()[1])
        if impute_input:
            # Remember that the dataset has zero mean.
            dataset_mean = tf.zeros_like(inputs_ph)
            x = tf.where(mask_missing_ph, dataset_mean, inputs_ph, name="ampute")
        else:
            x = inputs_ph
        x = tf.nn.dropout(x, keep_prob=FLAGS.input_dropout)

def build_input_machinery(data, mask_missing, num_epochs, shuffle, batch_size, name):
    with tf.variable_scope(name):
        q = tf.train.slice_input_producer(
                [data, mask_missing], num_epochs=num_epochs,
                shuffle=shuffle,
                capacity=batch_size*3)
        return tf.train.batch(q, batch_size)

def main(_):
    mask, original = pu.load('dataset.pkl.gz')
    n_features = original.shape[1]
    data_idx = np.arange(original.shape[0])
    np.random.shuffle(data_idx)
    train_len = int(len(data_idx)*0.7)
    train_idx, test_idx = data_idx[:train_len], data_idx[train_len:]

    train = build_input_machinery(original[train_idx,:].astype(np.float32),
                                 mask[train_idx,:],
                                 FLAGS.num_epochs, True, FLAGS.batch_size, "train")

    test = [tf.constant(original[test_idx,:].astype(np.float32), dtype=tf.float32),
            tf.constant(mask[test_idx,:], dtype=tf.bool)]
    inputs_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
    mask_ph = tf.placeholder(dtype=tf.bool, shape=[None, n_features])
    assert FLAGS.log_dir
    model = autoencoder_impute
    #model = autoencoder_dropout

    train_preds, name = model(*train)
    train_loss = mse_sum_train(train[1], train[0], train_preds)
    validation_preds, _ = model(*test, reuse=True, trainable=False)
    validation_loss = tf_rmse_sum_test(test[1], test[0], validation_preds)
    final_preds, _ = model(inputs_ph, mask_ph, impute_input=False, reuse=True, trainable=False)
    with tf.variable_scope("rmse"):
        train_summary = tf.summary.scalar("training/{:s}".format(name), train_loss)
        validation_summary = tf.summary.scalar("validation/{:s}".format(name), validation_loss)
    summary = tf.summary.merge([train_summary, validation_summary])
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(train_loss)

    imputed = []
    imputed_mcmc = []
    #for i in range(5):
    #    with tf.Session() as sess:
    #        train_f(i, sess, FLAGS.log_dir, train_op, summary, [validation_loss])
    #        imputed.append(sess.run(final_preds))
    saver = tf.train.Saver(max_to_keep=0)
    with tf.Session() as sess:
        #step = train_f(0, sess, FLAGS.log_dir, train_op, summary, [validation_loss])
        #saver.save(sess, FLAGS.log_dir+'/ckpt', global_step=step)
        saver.restore(sess, 'logs/all_dropout/ckpt-12900')
        samples = np.random.normal(size=original.shape)
        for i in range(5):
            samples = np.zeros_like(original)
            samples[~mask] = original[~mask]
            imputed.append(samples)

            samples = np.random.normal(size=original.shape)
            for i in range(100):
                samples[~mask] = original[~mask]
                samples = sess.run(final_preds, {mask_ph: mask, inputs_ph: samples})
            imputed_mcmc.append(samples)

    pu.dump(imputed, "autoencoder_iterate.pkl.gz")
    pu.dump(imputed_mcmc, "autoencoder_iterate_mcmc.pkl.gz")

def train_f(i, sess, log_dir, train_op, summary, validation):
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    summary_writer = tf.summary.FileWriter('{:s}/{:d}'.format(log_dir, i), graph=sess.graph)
    min_l = np.inf
    op_list = [train_op, summary] + validation
    try:
        for step in tqdm(itertools.count(1)):
            if coord.should_stop():
                break
            if step%100 == 0:
                _, s, l, *_ = sess.run(op_list)
                summary_writer.add_summary(s, step)
                if l < min_l:
                    min_l = l
                    patience = 0
                    print("new min_l:", min_l)
                else:
                    patience += 1
                    print("patience...")
                    if patience >= FLAGS.patience:
                        break
            else:
                sess.run(train_op)
    except tf.errors.OutOfRangeError:
        pass

    coord.request_stop()
    coord.join(queue_threads)
    return step

if __name__ == '__main__':
    tf.app.run()
