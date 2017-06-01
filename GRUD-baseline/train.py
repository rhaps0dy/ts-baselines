import tensorflow as tf
import pickle_utils as pu
import itertools as it
import os.path

from read_tfrecords import build_input_machinery
import model

flags = tf.app.flags
flags.DEFINE_string('command', 'train', 'What to do [train, test]')
flags.DEFINE_string('log_dir', './logs/', 'Base directory for logs')
flags.DEFINE_integer('min_after_dequeue', 5000,
                     'Minimum number of examples to draw from in the RandomShuffleQueue')
flags.DEFINE_integer('n_queue_threads', 2, 'number of queue threads')
flags.DEFINE_integer('batch_size', 64, 'batch size for training')
flags.DEFINE_integer('num_epochs', 1000, 'number of training epochs')
flags.DEFINE_integer('hidden_units', 100, 'Number of hidden units per LSTM layer')
flags.DEFINE_integer('hidden_layers', 1, 'Number of hidden LSTM layers')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
flags.DEFINE_float('dropout', 0.5, 'probability of keeping a neuron on')
flags.DEFINE_boolean('layer_norm', True, 'Whether to use Layer Normalisation')
flags.DEFINE_string('model', 'GRUD', 'the model to use')
del flags
FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    training = build_input_machinery(["dataset/train_0.tfrecords"], True,
        FLAGS.num_epochs, FLAGS.batch_size, FLAGS.min_after_dequeue,
        FLAGS.n_queue_threads)
    #validation = build_input_machinery(["hv_validation.tfrecords"], False,
    #    FLAGS.num_epochs, FLAGS.batch_size, FLAGS.min_after_dequeue,
    #    FLAGS.n_queue_threads)
    input_means = pu.load('dataset/means.pkl.gz')
    number_of_categories = pu.load('dataset/number_of_categories.pkl.gz')
    _, _, numerical_headers, categorical_headers, treatments_headers = \
        pu.load('dataset/small.pkl.gz')

    Model = getattr(model, FLAGS.model)
    m = Model(num_units=FLAGS.hidden_units,
              num_layers=FLAGS.hidden_layers,
              inputs_dict=training,
              input_means_dict=input_means,
              number_of_categories=number_of_categories,
              categorical_headers=categorical_headers,
              default_batch_size=FLAGS.batch_size,
              layer_norm=FLAGS.layer_norm)
    global_step = tf.train.get_or_create_global_step()
    train_op = (tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                .minimize(m['loss'], global_step=global_step))


    saver = tf.train.Saver(max_to_keep=0)
    with tf.variable_scope('summaries'):
        loss_ph = tf.placeholder(shape=[], dtype=tf.float32)
        training_summary = tf.summary.scalar('training/loss', loss_ph)
    #summary_op = tf.merge_all_summaries()

    sv = tf.train.Supervisor(is_chief=True,
                             logdir=FLAGS.log_dir,
                             summary_op=None,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    with open(os.path.join(FLAGS.log_dir, 'flags.txt'), 'w') as f:
        f.write(repr(FLAGS.__flags)+'\n')

    normal_ops = [train_op, m['loss']]
    summary_ops = [train_op, m['loss'], training_summary]
    loss_avg = 0.0
    d = {m['keep_prob']: FLAGS.dropout, loss_ph: 0.0}
    with sv.managed_session() as sess:
        sess.run(tf.get_collection('make_embeddings_nan'))
        for step in it.count(1):
            if sv.should_stop():
                break
            if step % 100 == 0:
                d[loss_ph] = loss_avg/100
                _, loss_avg, summ = sess.run(summary_ops, d)
                sv.summary_computed(sess, summ)
            else:
                _, loss = sess.run(normal_ops, d)
                loss_avg += loss
        sv.stop()
