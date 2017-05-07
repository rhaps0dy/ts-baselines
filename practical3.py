#!/usr/bin/env python3

import tensorflow as tf
import logging
import math
from collections import defaultdict

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

from utils import *

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

def main(_):
    log_dir, save_model_file, load_file = get_relevant_directories()
    training_data, validation_data, test_data = \
        split_dataset(input_texts, input_targets)
    del input_texts, input_targets
    word_index, embedding_matrix = generate_word_embeddings(
        list(t[0] for t in training_data))
    assert embedding_matrix.any()
    word_index = defaultdict(lambda: word_index['UNK'], word_index)
    reverse_word_index = dict((v, k) for k, v in word_index.items())

    training_data = numerise_dataset(training_data, word_index)
    training = models.TrainingManager(training_data, FLAGS.batch_size, FLAGS.bptt_len)
    training_test = models.TestManager(training_data[:250], FLAGS.batch_size,
                                       FLAGS.bptt_len)
    validation = models.TestManager(numerise_dataset(validation_data, word_index),
                                    FLAGS.batch_size, FLAGS.bptt_len)
    test = models.TestManager(numerise_dataset(test_data, word_index),
                              FLAGS.batch_size, FLAGS.bptt_len)
    del training_data, validation_data, test_data

    log.info("Building model...")

    sess = tf.Session()

    m = getattr(models, FLAGS.model)(num_units=FLAGS.hidden_units,
                                     num_layers=FLAGS.hidden_layers,
                                     embedding_matrix=embedding_matrix,
                                     training_keep_prob=FLAGS.dropout,
                                     bptt_length=FLAGS.bptt_len,
                                     softmax_samples=FLAGS.softmax_samples,
                                     batch_size=FLAGS.batch_size)
    del embedding_matrix

    train_step = m.train_step(getattr(tf.train, FLAGS.optimizer))

    # Model checkpoints and graphs
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    if load_file:
        saver.restore(sess, load_file)
        log.info("Loaded model from file %s" % load_file)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    training_summary = tf.summary.scalar('training/loss', m.loss)
    perplexity_ph = tf.placeholder(tf.float32, shape=[])
    training_perp_summary = tf.summary.scalar('training/entropy', perplexity_ph)
    validation_summary = tf.summary.scalar('validation/entropy', perplexity_ph)

    JOIN_CHAR = " " if FLAGS.input_type == "word" else ""
    if FLAGS.command == 'train':
        learning_rate = FLAGS.learning_rate
        log.info("Each epoch has {:d} steps.".format(training.epoch_steps))
        for epoch in range(1, FLAGS.max_epochs+1):
            log.info("Training epoch %d..." % epoch)
            training.new_training_permutation()
            m.new_epoch()
            for i, training_example in enumerate(training):
                summary_t = (epoch-1) * training.epoch_steps + i
                feed_dict = m.training_feed_dict(training_example, learning_rate)
                if i % 10 == 9:
                    log.info("Running example {:d}".format(i+1))
                result = sess.run([training_summary, train_step] +
                                  m.next_state, feed_dict)
                summary_writer.add_summary(result[0], summary_t)
                if i % 1000 == 999:
                    def compute_perplexity(test_mgr, summary_op):
                        perplexity = m.compute_entropy(sess, test_mgr)
                        summary, = sess.run([summary_op],
                                            {perplexity_ph: perplexity})
                        summary_writer.add_summary(summary, summary_t)
                        return perplexity

                    perplexity = compute_perplexity(training_test, training_perp_summary)
                    log.info("Training entropy is {:.4f}".format(perplexity))
                    perplexity = compute_perplexity(validation, validation_summary)
                    log.info("Entropy is {:.4f}".format(perplexity))
                    if math.isnan(perplexity) or math.isinf(perplexity):
                        import pdb
                        pdb.set_trace()

                    log.info("Generating 10 phrases:")
                    cur_words = [word_index['START']]*10
                    ws = [[] for _ in range(10)]
                    m.new_epoch(batch_size=10)
                    for w in range(20 if FLAGS.input_type == "word" else 100):
                        d = m.test_feed_dict(([[cw]+[0]*FLAGS.bptt_len for cw
                                               in cur_words], [1]*10))
                        draw_result = sess.run([m.prediction_sample] + m.next_state, d)
                        m.state = draw_result[1:]
                        cur_words = draw_result[0]
                        for j in range(len(cur_words)):
                            ws[j].append(reverse_word_index[cur_words[j]])
                    for w in ws:
                        log.info("generated_phrase: " + JOIN_CHAR.join(w))
                    save_path = saver.save(sess, save_model_file, global_step=summary_t)
                    log.info("Model saved in file: {:s}, validation entropy {:.4f}"
                             .format(save_path, perplexity))
                    learning_rate = max(learning_rate*FLAGS.learning_rate_decay,
                                        FLAGS.min_learning_rate)
                    log.info("New learning rate is {:f}".format(learning_rate))
                m.state = result[2:]
    elif FLAGS.command == 'compute_entropy':
        ent = m.compute_entropy(sess, validation)
        print("The entropy is", ent)
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
    elif FLAGS.command == 'print_stuff':
        training.new_training_permutation()
        m.new_epoch()
        for i, example in enumerate(training_test):
            print(" ".join(reverse_word_index[i] for i in example[0][0]))
            feed_dict = m.test_feed_dict(example)
            predictions, entropy_sum, nsc, nsh = sess.run([m.prediction_max, m.entropy_sum,
                m.next_state_c, m.next_state_h], feed_dict)
            print("Entropy is", entropy_sum/len(predictions))
            print(" ".join(reverse_word_index[i] for i in predictions))
            input()


    sess.close()

if __name__ == '__main__':
    tf.app.run()
