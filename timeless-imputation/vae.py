import tensorflow as tf
import pickle_utils as pu
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
from neural_networks import layer, build_input_machinery, tf_rmse_sum_test
import bb_alpha_inputs as bba

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('log_dir', '', 'Directory to log to')
    flags.DEFINE_string('command', 'train', 'command to perform [train,impute]')
    flags.DEFINE_string('dataset', None, 'Dataset to read')
    flags.DEFINE_integer('batch_size', 64, 'batch size for training')
    flags.DEFINE_integer('num_epochs', 1000, 'number of training epochs')
    flags.DEFINE_integer('increment', 7, 'number of training epochs')
    flags.DEFINE_integer('num_q_layers', 1, 'number of hidden layers in q(z|x)')
    flags.DEFINE_integer('patience', 1000000000, 'number of training epochs')
    flags.DEFINE_integer('num_samples', 10, 'number of training epochs')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_float('input_dropout', 0.5, 'droppo')
    del flags
FLAGS = tf.app.flags.FLAGS

@bba.add_variable_scope()
def gaussian_nn(inputs, layer_sizes, nlin, num_samples, trainable):
    num_features = int(inputs.get_shape()[-1])
    if len(inputs.get_shape()) > 2:
        num_samples = int(inputs.get_shape()[1])
    else:
        num_samples = 1
    layer_sizes = layer_sizes.copy()
    latent_dims = layer_sizes[-1]
    layer_sizes[-1] *= 2

    _z = tf.reshape(inputs, [-1, num_features])
    for i, size in enumerate(layer_sizes):
        name = "layer_{:d}".format(i)
        _z = layer(_z, size, name, trainable=trainable,
                  nlin=(nlin if i<len(layer_sizes)-1 else None))
        _z = tf.check_numerics(_z, name)
    latent_params_z = tf.reshape(_z, [-1, num_samples, latent_dims*2])

    if num_samples > 1:
        latent_params_z = tf.reduce_mean(latent_params_z, axis=1, keep_dims=True)

    z_mu = latent_params_z[..., :latent_dims]
    log_z_sig2 = latent_params_z[..., latent_dims:]
    z_sig2 = tf.exp(log_z_sig2)
    z_sig = tf.sqrt(z_sig2)
    z_sig = tf.check_numerics(z_sig, "z_sig")
    z = tf.random_normal([tf.shape(inputs)[0], num_samples, latent_dims])*z_sig + z_mu
    z = tf.check_numerics(z, "z")

    return z, log_z_sig2, z_sig2, z_sig, z_mu

@bba.add_variable_scope(name="vae")
def variational_autoencoder(inputs, mask_missing, M, num_samples=11,
                            n_latent_samples=11, nlin=tf.nn.tanh,
                            trainable=True):
    name = "vae"
    _inputs = tf.tile(tf.expand_dims(inputs, 1), [1, num_samples, 1])
    _mask_one = tf.expand_dims(mask_missing, 1)
    _mask = tf.tile(_mask_one, [1, num_samples, 1])

    num_features = int(_inputs.get_shape()[2])
    layer_sizes = list(num_features+FLAGS.increment*i for i in range(FLAGS.num_q_layers+2))

    latent_samples = tf.random_normal([tf.shape(inputs)[0], num_samples,
                                       layer_sizes[-1]], dtype=tf.float32)
    gaussian_impute, *_ = gaussian_nn(latent_samples, layer_sizes[-2::-1], nlin=nlin,
                                      num_samples=FLAGS.num_samples,
                                      trainable=trainable,
                                      name="p_x_z")

    #gaussian_impute = tf.random_normal(tf.shape(_inputs), dtype=tf.float32)
    #gaussian_impute = tf.zeros_like(_inputs)
    missing_inputs = tf.where(_mask, gaussian_impute, _inputs)

    z, log_z_sig2, z_sig2, z_sig, z_mu = gaussian_nn(missing_inputs, layer_sizes[1:],
                                            nlin=nlin,
                                            num_samples=FLAGS.num_samples,
                                            trainable=trainable,
                                            name="q_z_x")
    x, log_x_sig2, x_sig2, x_sig, x_mu = gaussian_nn(z, layer_sizes[-2::-1],
                                            nlin=nlin,
                                            num_samples=FLAGS.num_samples,
                                            trainable=trainable,
                                            name="p_x_z",
                                            reuse=True)
    log_x_sig = log_x_sig2 / 2.

    with tf.variable_scope("loss"):
        q_lik = .5*tf.reduce_sum(1 + log_z_sig2 - tf.square(z_mu) - z_sig2,
                                 axis=2, name="q_likelihood")
        q_lik = tf.squeeze(q_lik, axis=1)
        q_lik = tf.check_numerics(q_lik, "q_lik")
        # _inputs is of size [.. num_samples ..] rather than [.. n_latent_samples ..]
        p_lik = bba.make_log_likelihood(x_mu, _inputs, log_x_sig2, x_sig2,
                                        condition=_mask, noise_condition=_mask_one)
        p_lik = tf.check_numerics(p_lik, "p_lik")
        loss = q_lik + tf.reduce_mean(p_lik, axis=1)
        loss = -M*tf.reduce_mean(loss)
        return tf.reduce_mean(x, axis=1), name, loss, (x_mu, log_x_sig2, x_sig2)

def main(_):
    mask, original = pu.load(FLAGS.dataset)
    print(mask.shape)
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
    model = variational_autoencoder


    train[0] = tf.check_numerics(train[0], "train_0")
    test[0] = tf.check_numerics(test[0], "test_0")
    _, name, train_loss, _ = model(*train,
                                M=len(original),
                                num_samples=FLAGS.num_samples,
                                n_latent_samples=FLAGS.num_samples)
    validation_preds, _, _, (x_mu, log_x_sig2, x_sig2) = model(
        *test, reuse=True, trainable=False, M=len(original),
         num_samples=FLAGS.num_samples, n_latent_samples=FLAGS.num_samples)
    validation_log_likelihood = tf.reduce_sum(bba.make_log_likelihood(
        tf.squeeze(x_mu, axis=1), test[0],
        tf.squeeze(log_x_sig2, axis=1), tf.squeeze(x_sig2, axis=1),
        condition=test[1]), axis=0)
    validation_rmse = tf_rmse_sum_test(test[1], test[0], validation_preds)
    final_preds, _, _, _ = model(inputs_ph, mask_ph, reuse=True,
                              M=len(original),
                              trainable=False, num_samples=FLAGS.num_samples,
                              n_latent_samples=FLAGS.num_samples)
    with tf.variable_scope("validation"):
        validation_rmse_summary = tf.summary.scalar("rmse", validation_rmse)
        validation_ll_summary = tf.summary.scalar("log_likelihood", validation_log_likelihood)
    with tf.variable_scope("training"):
        train_loss_summary = tf.summary.scalar("loss", train_loss)
    summary = tf.summary.merge([validation_rmse_summary, validation_ll_summary])
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        train_loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=0)
    if FLAGS.command == 'train':
        sv = tf.train.Supervisor(is_chief=True,
                                logdir=FLAGS.log_dir,
                                summary_op=None,
                                saver=saver,
                                global_step=global_step,
                                save_model_secs=600)

        def validate_current(sess, summary_writer):
            summ, step, *_ = sess.run([summary, global_step, validation_rmse,
                                    validation_log_likelihood])
            summary_writer.add_summary(summ, step)

        with sv.managed_session() as sess:
            summary_writer = tf.summary.FileWriter('{:s}'.format(FLAGS.log_dir),
                                                graph=sess.graph)
            sv.loop(300, validate_current, args=(sess, summary_writer))
            for step in tqdm(itertools.count(1)):
                if sv.should_stop():
                    break
                if step % 100 == 0:
                    _, summ, g_step = sess.run([train_loss_summary, train_op, global_step])
                    summary_writer.add_summary(summ, g_step)
                else:
                    sess.run(train_op)
            sv.stop()
    elif FLAGS.command == 'impute':
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))

            imputed = []
            imputed_mcmc = []
            # Output samples
            samples = np.random.normal(size=original.shape)

            for i in tqdm(range(5)):
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                samples = np.zeros_like(original)
                samples[~mask] = original[~mask]
                samples = sess.run(final_preds, {mask_ph: mask, inputs_ph: samples})
                samples[~mask] = original[~mask]
                imputed.append(samples)

                for i in range(10):
                    samples[~mask] = original[~mask]
                    samples = sess.run(final_preds, {mask_ph: mask, inputs_ph: samples})
                samples[~mask] = original[~mask]
                imputed_mcmc.append(samples)

            pu.dump(imputed, "vae_iterate.pkl.gz")
            pu.dump(imputed_mcmc, "vae_iterate_mcmc.pkl.gz")

if __name__ == '__main__':
    tf.app.run()
