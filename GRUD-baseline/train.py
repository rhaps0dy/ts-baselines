import tensorflow as tf
import pickle_utils as pu
import itertools as it
import os.path

from read_tfrecords import build_input_machinery
import model

flags = tf.app.flags
flags.DEFINE_string('command', 'train', 'What to do [train, validate, profile_training]')
flags.DEFINE_string('load_file', '', 'What file to load for validating?')
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

def main(_):
    shuffle = True
    if FLAGS.command == 'validate':
        assert FLAGS.num_epochs == 1, "We only want to go through the sets once"
        shuffle = False
        validation = build_input_machinery(["dataset/validation_0.tfrecords"],
            False, FLAGS.num_epochs, FLAGS.batch_size, FLAGS.min_after_dequeue,
            FLAGS.n_queue_threads)

    training = build_input_machinery(["dataset/train_0.tfrecords"], shuffle,
        FLAGS.num_epochs, FLAGS.batch_size, FLAGS.min_after_dequeue,
        FLAGS.n_queue_threads)

    input_means = pu.load('dataset/means.pkl.gz')
    number_of_categories = pu.load('dataset/number_of_categories.pkl.gz')
    _, _, numerical_headers, categorical_headers, treatments_headers = \
        pu.load('dataset/small.pkl.gz')

    def build_model(inputs, reuse=None):
        Model = getattr(model, FLAGS.model)
        with tf.variable_scope(FLAGS.model, reuse=reuse):
            m = Model(num_units=FLAGS.hidden_units,
                    num_layers=FLAGS.hidden_layers,
                    inputs_dict=inputs,
                    input_means_dict=input_means,
                    number_of_categories=number_of_categories,
                    categorical_headers=categorical_headers,
                    default_batch_size=FLAGS.batch_size,
                    layer_norm=FLAGS.layer_norm)
            return m

    m = build_model(training)

    if FLAGS.command == 'validate':
        v_m = build_model(validation, reuse=True)
        hours_before = [24, 12, 8, 4, 0]
        with tf.variable_scope('metrics'):
            #metrics = model.ventilation_risk_score(training, m, hours_before)
            v_metrics = model.ventilation_risk_score(validation, v_m, hours_before)
            #summaries = []
            #for name, metrics in [('training', metrics), ('validation', v_metrics)]:
            #    for h, metric in zip(hours_before, metrics):
            #        summaries.append(
            #            tf.summary.scalar('{:s}/{:d}_hours'.format(name, h), metric))
            #validation_summary = tf.summary.merge(summaries)


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

    if FLAGS.command == 'profile_training':
        def do_step(step):
            run_metadata = tf.RunMetadata()
            _ = sess.run(
                        [train_op, m['loss'], m['flat_risk_score']],
                        feed_dict={m['keep_prob']: FLAGS.dropout},
                        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        run_metadata=run_metadata)
            from tensorflow.python.client import timeline
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())
            return True

    elif FLAGS.command == 'train':
        with open(os.path.join(FLAGS.log_dir, 'flags.txt'), 'w') as f:
            f.write(repr(FLAGS.__flags)+'\n')

        normal_ops = [train_op, m['loss']]
        summary_ops = [train_op, m['loss'], training_summary]
        d = {m['keep_prob']: FLAGS.dropout, loss_ph: 0.0}

        def do_step(step):
            if step == 50:
                run_metadata = tf.RunMetadata()
                _ = sess.run(
                            [train_op, m['loss'], m['flat_risk_score']],
                            feed_dict={m['keep_prob']: FLAGS.dropout},
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=run_metadata)
                from tensorflow.python.client import timeline
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open('timeline_50.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format())
                return True
            if step % 100 == 0:
                d[loss_ph] /= 100
                _, d[loss_ph], summ = sess.run(summary_ops, d)
                sv.summary_computed(sess, summ)
            else:
                _, loss = sess.run(normal_ops, d)
                d[loss_ph] += loss

    elif FLAGS.command == 'validate':

        d = {m['keep_prob']: 1.0, v_m['keep_prob']: 1.0}
        #l = [m['loss'], v_m['loss'], validation_summary]
        def do_step(step):
            #train_loss, validation_loss, summary = sess.run(l, d)
            #sv.summary_computed(sess, summary)
            #d[loss_ph] = train_loss
            #summary = sess.run(loss_ph, d)
            #sv.summary_computed(sess, summary)
            a = sess.run(v_metrics[0], d)
            b = v_metrics[1](*a)
    else:
        raise ValueError("Unknown command: `{:s}`".format(FLAGS.command))

    with sv.managed_session() as sess:
        sess.run(tf.get_collection('make_embeddings_nan'))
        for step in it.count(1):
            if sv.should_stop() or do_step(step):
                break
        sv.stop()

if __name__ == '__main__':
    tf.app.run()
