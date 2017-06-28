import tensorflow as tf
import pickle_utils as pu
import numpy as np
import itertools as it
import os
from tqdm import tqdm

import bb_alpha_inputs as bba
from imputation_read_tfrecords import build_input_machinery
import train

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('command', 'train',
                        'What to do [train, validate, profile_training]')
    flags.DEFINE_string('load_file', '', 'What file to load for validating?')
    flags.DEFINE_integer('batch_size', 64, 'batch size for training')
    flags.DEFINE_integer('num_samples', 16, 'samples to estimate expectation')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
    flags.DEFINE_string('layer_sizes', '[64, 64]', 'layer sizes')
    flags.DEFINE_string('training_set',
                        '../clean_ventilation/dataset/train_0.tfrecords-imputation',
                        'Location of training set')
    flags.DEFINE_string('validation_set',
                        '../clean_ventilation/dataset/validation_0.tfrecords-imputation',
                        'Location of validation set')
    flags.DEFINE_string('log_dir', None, 'Location of ')
    flags.DEFINE_integer('patience', 20, 'Number of epochs to wait if'
                         'validation log-likelihood does not increase')
    flags.DEFINE_integer('num_epochs', 1000, 'number of training epochs')
    flags.DEFINE_integer('min_after_dequeue', 5000,
                        "Minimum number of examples to draw from in the "
                         "RandomShuffleQueue")
    flags.DEFINE_integer('n_queue_threads', 2, 'number of queue threads')
    del flags
FLAGS = tf.app.flags.FLAGS

def main(_):
    assert FLAGS.log_dir is not None
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9/2
    shuffle = True
    dataset_dir = os.path.dirname(FLAGS.training_set)
    feature_numbers = pu.load(os.path.join(dataset_dir, 'feature_numbers.pkl.gz'))
    print(feature_numbers)
    if FLAGS.command == 'validate':
        assert FLAGS.load_file, "Must indicate a file to load"
        assert os.path.isfile(FLAGS.load_file+'.index')
        assert os.path.dirname(FLAGS.load_file) == FLAGS.log_dir.rstrip("/"), \
            "File to load must be in log_dir"
        assert FLAGS.num_epochs == 1, "We only want to go through the sets once"
        shuffle = False
        validation = build_input_machinery([FLAGS.validation_set],
                                           feature_numbers, False,
                                           FLAGS.num_epochs, FLAGS.batch_size,
                                           FLAGS.min_after_dequeue,
                                           FLAGS.n_queue_threads)
    training = build_input_machinery([FLAGS.training_set],
                                    feature_numbers, shuffle,
                                    FLAGS.num_epochs, FLAGS.batch_size,
                                    FLAGS.min_after_dequeue,
                                    FLAGS.n_queue_threads)

    whiten = pu.load(os.path.join(dataset_dir, 'whiten_imputation.pkl.gz'))
    means_X = np.concatenate(list(whiten['values'][k] / whiten['counts'][k]
                                  for k in ['num_forward', 'num_ts']),
                             axis=0).astype(np.float32)
    stddevs_X = np.concatenate(list(whiten['stddevs'][k] for k in ['num_forward', 'num_ts']),
                               axis=0).astype(np.float32)
    means_y = (whiten['values']['num_labels'] / whiten['counts']['num_labels']).astype(np.float32)
    stddevs_y = whiten['stddevs']['num_labels'].astype(np.float32)

    def build_model(inputs_dict, reuse=None):
        with tf.variable_scope("preprocessing", reuse=reuse):
            inputs = tf.concat([inputs_dict['num_forward'], inputs_dict['num_ts']], axis=1)
            inputs = tf.check_numerics(inputs, "inputs")
        m = bba.model(inputs,
                      inputs_dict['num_labels'],
                      N=whiten['total_n'],
                      num_samples=FLAGS.num_samples,
                      layer_sizes=eval(FLAGS.layer_sizes),
                      alpha=0.5,
                      trainable=True, # This controls which output we have
                      include_samples=False,
                      mean_X=means_X,
                      mean_y=means_y,
                      std_X=stddevs_X,
                      std_y=stddevs_y,
                      name="num_all",
                      reuse=reuse)
        return m

    m = build_model(training)
    if FLAGS.command == 'validate':
        v_m = build_model(validation, reuse=True)
        mse = {}
        mse_ph = {}
        log_likelihood = {}
        log_likelihood_ph = {}
        actual_batch_size = {}
        _summaries = []
        with tf.variable_scope('metrics'):
            for name, mdl, inputs_dict in [('training', m, training), ('validation', v_m, validation)]:
                distances = tf.reduce_sum(tf.squared_difference(
                    mdl['mean_prediction'], inputs_dict['num_labels']), axis=1)
                mse[name] = tf.reduce_mean(distances, axis=0)
                mse_ph[name] = tf.placeholder(dtype=tf.float32, shape=[], name="{:s}/mse".format(name))
                _summaries.append(tf.summary.scalar('{:s}/mse'.format(name), mse_ph[name]))
                log_likelihood[name] = tf.reduce_sum(mdl['log_likelihood'], axis=0)
                log_likelihood_ph[name] = tf.placeholder(dtype=tf.float32, shape=[], name="{:s}/log_likelihood".format(name))
                _summaries.append(tf.summary.scalar('{:s}/log_likelihood'.format(name), log_likelihood_ph[name]))
                actual_batch_size[name] = tf.shape(mdl['mean_prediction'])[0]
            validation_summaries = tf.summary.merge(_summaries)
            del _summaries

    global_step = tf.train.get_or_create_global_step()
    train_op = (tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                .minimize(m['energy'], global_step=global_step))

    saver = tf.train.Saver(max_to_keep=0)
    with tf.variable_scope('metrics'):
        loss_ph = tf.placeholder(shape=[], dtype=tf.float32)
        training_summary = tf.summary.scalar('training/loss', loss_ph)

    if FLAGS.command == 'train':
        sv = tf.train.Supervisor(is_chief=True,
                                logdir=FLAGS.log_dir,
                                summary_op=None,
                                saver=saver,
                                global_step=global_step,
                                save_model_secs=600)

        # Save invocation flags to log_dir
        with open(os.path.join(FLAGS.log_dir, 'flags.txt'), 'w') as f:
            f.write(repr(FLAGS.__flags)+'\n')

        normal_ops = [train_op, m['energy']]
        summary_ops = [train_op, m['energy'], training_summary]
        d = {loss_ph: 0.0}

        validate_checkpoint_persist = {}
        with sv.managed_session(config=config) as sess:
            sv.loop(300, train.validate_checkpoint, args=(validate_checkpoint_persist,))
            for step in tqdm(it.count(1)):
                if sv.should_stop():
                    break
                if step % 100 == 0:
                    d[loss_ph] /= 100
                    _, d[loss_ph], summ = sess.run(summary_ops, d)
                    sv.summary_computed(sess, summ)
                else:
                    _, loss = sess.run(normal_ops)
                    d[loss_ph] += loss
            sv.stop()

    elif FLAGS.command == 'validate':
        print("DOING VALIDATE")
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver.restore(sess, FLAGS.load_file)

            mse_values = {'training': 0.0, 'validation': 0.0}
            log_likelihood_values = {'training': 0.0, 'validation': 0.0}
            def add_to_means_preds(name, n_steps=5):
                try:
                    for step in range(n_steps):
                        if coord.should_stop():
                            break
                        _mse, _ll, _bs = sess.run([mse[name], log_likelihood[name],
                                                   actual_batch_size[name]])
                        assert _bs == FLAGS.batch_size, "YOU MUST ACCOUNT FOR DIFFERENT-SIZED BATCHES IN MEANS"
                        mse_values[name] += _mse/n_steps
                        log_likelihood_values[name] += _ll
                except tf.errors.OutOfRangeError:
                    pass

            for name in mse_values.keys():
                add_to_means_preds(name)

            coord.request_stop()
            coord.join(queue_threads)

            summary_writer = tf.summary.FileWriter(FLAGS.log_dir)

            feed_dict = {}
            for name in mse_values.keys():
                feed_dict[mse_ph[name]] = mse_values[name]
                feed_dict[log_likelihood_ph[name]] = log_likelihood_values[name]
            s, t = sess.run([validation_summaries, global_step], feed_dict)
            summary_writer.add_summary(s, t)

    else:
        raise ValueError("Unknown command: `{:s}`".format(FLAGS.command))

if __name__ == '__main__':
    tf.app.run()
