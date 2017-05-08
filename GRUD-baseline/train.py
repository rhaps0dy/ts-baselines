#!/usr/bin/env python3

import tensorflow as tf
import logging
import math
import numpy as np
from collections import defaultdict
from pkl_utils import *
import model

flags = tf.app.flags

flags.DEFINE_boolean('load_latest', False, 'Whether to load the last model')
flags.DEFINE_string('load_latest_from', '', 'Folder to load the model from')
flags.DEFINE_string('load_from', '', 'File to load the model from')
flags.DEFINE_boolean('log_to_stdout', True, 'Whether to output the python log '
                     'to stdout, or to a file')
flags.DEFINE_string('out_file', 'out', 'the file to output test scores / sample phrases to')
flags.DEFINE_string('command', 'train', 'What to do [train, test]')
flags.DEFINE_string('log_dir', './logs/', 'Base directory for logs')
flags.DEFINE_string('model_dir', './model/', 'Base directory for model')
flags.DEFINE_integer('batch_size', 64, 'batch size for training')
flags.DEFINE_integer('max_epochs', 1000, 'maximum training epochs')
flags.DEFINE_integer('hidden_units', 100, 'Number of hidden units per LSTM layer')
flags.DEFINE_integer('hidden_layers', 1, 'Number of hidden LSTM layers')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
flags.DEFINE_float('dropout', 0.5, 'probability of keeping a neuron on')
flags.DEFINE_string('optimizer', 'AdamOptimizer', 'the optimizer to use')
flags.DEFINE_string('model', 'GRUD', 'the model to use')
flags.DEFINE_string('log_level', 'INFO', 'logging level')

FILENAME_FLAGS = ['learning_rate', 'batch_size', 'hidden_units',
                  'hidden_layers', 'dropout']

FLAGS = tf.app.flags.FLAGS
log = logging.getLogger(__name__)

def get_relevant_directories():
    # Give the model a nice name in TensorBoard
    current_flags = []
    for flag in FILENAME_FLAGS:
        current_flags.append('{}={}'.format(flag, getattr(FLAGS, flag)))
    _log_dir = FLAGS.log_dir = os.path.join(FLAGS.log_dir, *current_flags)
    _model_dir = FLAGS.model_dir = os.path.join(FLAGS.model_dir, *current_flags)
    i=0
    while os.path.exists(FLAGS.log_dir):
        i += 1
        FLAGS.log_dir=('{}/{}'.format(_log_dir, i))
    FLAGS.log_dir=('{}/{}'.format(_log_dir, i))
    log_file = os.path.join(FLAGS.log_dir, 'console_log.txt')
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    basicConfigKwargs = {'level': getattr(logging, FLAGS.log_level.upper()),
                         'format': '%(asctime)s %(name)s %(message)s'}
    if not FLAGS.log_to_stdout:
        basicConfigKwargs['filename'] = log_file
    logging.basicConfig(**basicConfigKwargs)
    save_model_file=('{}/{}/ckpt'.format(_model_dir, i))
    save_model_dir=('{}/{}'.format(_model_dir, i))
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    if FLAGS.load_latest or FLAGS.load_latest_from:
        if FLAGS.load_latest:
            load_dir=('{}/{}'.format(_model_dir, i-1))
        else:
            load_dir=FLAGS.load_latest_from
        load_file = tf.train.latest_checkpoint(load_dir)
        if load_file is None:
            log.error("No checkpoint found!")
            exit(1)
    elif FLAGS.load_from:
        load_file = FLAGS.load_from
    else:
        load_file = None
    return FLAGS.log_dir, save_model_file, load_file

@memoize_pickle('split_dataset.pkl')
def split_dataset(fname):
    input_means, data = load_pickle(fname)
    ds = []
    for _, _, label, dt, X in data:
        ds.append((X, dt, len(X), label))
    np.random.shuffle(ds)
    test_len = int(len(ds) * .2)
    vali_len = int(len(ds) * .2)
    i1 = -test_len-vali_len
    i2 = -test_len
    return input_means, (ds[:i1], ds[i1:i2], ds[i2:])

def batch_generator(dataset, batch_size):
    return (list(zip(*dataset[i:i+batch_size]))
            for i in range(0, len(dataset), batch_size))

def main(_):
    log_dir, save_model_file, load_file = get_relevant_directories()
    input_means, (training_data, validation_data, test_data) = \
        split_dataset('../../mimic-clean/48h_training_examples.pkl')

    log.info("Building model...")

    sess = tf.Session()

    m = getattr(model, FLAGS.model)(num_units=FLAGS.hidden_units,
                                     num_layers=FLAGS.hidden_layers,
                                     input_means=input_means,
                                     training_keep_prob=FLAGS.dropout,
                                     bptt_length=training_data[0][2],
                                     batch_size=FLAGS.batch_size,
                                     n_classes=2)
    train_step = m.train_step()

    # Model checkpoints and graphs
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    if load_file:
        saver.restore(sess, load_file)
        log.info("Loaded model from file %s" % load_file)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    training_summary = tf.summary.scalar('training/loss', m.loss)
    loss_ph = tf.placeholder(tf.float32, shape=[])
    validation_summary = tf.summary.scalar('validation/loss', loss_ph)

    if FLAGS.command == 'train':
        learning_rate = FLAGS.learning_rate
        epoch_steps = (len(training_data)+FLAGS.batch_size-1) // FLAGS.batch_size
        log.info("Each epoch has {:d} steps.".format(epoch_steps))
        for epoch in range(1, FLAGS.max_epochs+1):
            log.info("Training epoch %d..." % epoch)
            np.random.shuffle(training_data)
            m.new_epoch()
            for i, training_example in enumerate(batch_generator(training_data, FLAGS.batch_size)):
                summary_t = (epoch-1) * epoch_steps + i
                feed_dict = m.feed_dict(training_example, learning_rate=FLAGS.learning_rate)
                if i % 10 == 9:
                    log.info("Running example {:d}".format(i+1))
                result = sess.run([training_summary, train_step], feed_dict)
                summary_writer.add_summary(result[0], summary_t)

            loss_mean = 0.0
            for i, validation_example in enumerate(batch_generator(validation_data, FLAGS.batch_size)):
                feed_dict = m.feed_dict(validation_example, training=False)
                loss, *_ = sess.run([m.loss], feed_dict)
                loss_mean += loss*len(validation_example[0])/len(validation_data)
            result = sess.run([validation_summary], {loss_ph: loss_mean})
            summary_writer.add_summary(result[0], summary_t)

            save_path = saver.save(sess, save_model_file, global_step=summary_t)
            log.info("Model saved in file: {:s}, validation loss {:.4f}"
                        .format(save_path, loss_mean))
    elif FLAGS.command == 'test':
        if not os.path.exists(FLAGS.out_file):
            with open(FLAGS.out_file, 'w') as f:
                f.write("Optimizer\nNonlinearity\nHidden units\nTraining\nValidation\nTest\n\n")
        with open(FLAGS.out_file, 'a') as f:
            f.write("{}\n{}\n{}\n".format(FLAGS.optimizer,
                FLAGS.nonlinearity, FLAGS.hidden_units))
            _acc, = sess.run([accuracy], feed_dict_from_data(training_data, dropout=1.0))
            f.write("{:.4f}\n".format(_acc))
            _acc, = sess.run([accuracy], feed_dict_from_data(validation_data, dropout=1.0))
            f.write("{:.4f}\n".format(_acc))
            _acc, = sess.run([accuracy], feed_dict_from_data(test_data, dropout=1.0))
            f.write("{:.4f}\n\n".format(_acc))
    else:
        raise ValueError(FLAGS.command)

    sess.close()

if __name__ == '__main__':
    tf.app.run()
