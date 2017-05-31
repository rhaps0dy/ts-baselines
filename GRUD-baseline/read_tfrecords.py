import tensorflow as tf
import pickle_utils as pu
import itertools as it
import model

flags = tf.app.flags
flags.DEFINE_string('command', 'train', 'What to do [train, test]')
flags.DEFINE_string('log_dir', './logs/', 'Base directory for logs')
flags.DEFINE_integer('min_after_dequeue', 5000,
                     'Minimum number of examples to draw from in the RandomShuffleQueue')
flags.DEFINE_integer('n_queue_threads', 2, 'number of queue threads')
flags.DEFINE_integer('batch_size', 64, 'batch size for training')
flags.DEFINE_integer('max_epochs', 1000, 'maximum training epochs')
flags.DEFINE_integer('hidden_units', 100, 'Number of hidden units per LSTM layer')
flags.DEFINE_integer('hidden_layers', 1, 'Number of hidden LSTM layers')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
flags.DEFINE_float('dropout', 0.5, 'probability of keeping a neuron on')
flags.DEFINE_boolean('layer_norm', True, 'Whether to use Layer Normalisation')
flags.DEFINE_string('model', 'GRUD', 'the model to use')
del flags
FLAGS = tf.app.flags.FLAGS

def read_and_decode(filename_queue):
    f_num = pu.load('feature_numbers.pkl.gz')
    reader = tf.TFRecordReader()
    _q, serialized_example = reader.read(filename_queue)
    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        context_features={
            'icustay_id': tf.FixedLenFeature([], tf.int64),
            'numerical_static': tf.FixedLenFeature(
                [f_num['numerical_static']], dtype=tf.float32),
            'categorical_static': tf.FixedLenFeature(
                [f_num['categorical_static']], dtype=tf.int64),
            'numerical_ts_dt': tf.FixedLenFeature(
                [f_num['numerical_ts']], dtype=tf.float32),
            'categorical_ts_dt': tf.FixedLenFeature(
                [f_num['categorical_ts']], dtype=tf.float32),

        }, sequence_features = {
           'time_until_label': tf.FixedLenSequenceFeature([], tf.float32),
           'label': tf.FixedLenSequenceFeature([], tf.int64),
           'numerical_ts': tf.FixedLenSequenceFeature(
               [f_num['numerical_ts']], dtype=tf.float32),
           'categorical_ts': tf.FixedLenSequenceFeature(
               [f_num['categorical_ts']], dtype=tf.int64),
           'treatments_ts': tf.FixedLenSequenceFeature(
               [f_num['treatments_ts']], dtype=tf.float32),
        })
    convert = {'icustay_id': tf.int32, 'label': tf.uint8,
               'categorical_static': tf.uint8, 'categorical_ts': tf.uint8}
    length = tf.shape[sequence['label']][0]
    return dict(map(
        lambda e: (e[0], tf.cast(e[1], convert[e[0]])) if e[0] in convert else e,
        it.chain(context.items(), sequence.items(), [("length", length)])))

def build_input_machinery(filenames, do_shuffle, num_epochs, batch_size,
        min_after_dequeue, n_queue_threads):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs)
    d = read_and_decode(filename_queue)
    d_keys, d_items = zip(*d.items())

    dtypes = list(a.dtype for a in d_items)
    shapes = list(a.get_shape() for a in d_items)

    if do_shuffle:
        shuffle_q = tf.RandomShuffleQueue(
            min_after_dequeue = min_after_dequeue,
            capacity = (min_after_dequeue +
                        (n_queue_threads+2)*batch_size),
            dtypes = dtypes)
        shuffle_op = shuffle_q.enqueue(d_items)
        qr = tf.train.QueueRunner(shuffle_q, [shuffle_op]*n_queue_threads)
        tf.train.add_queue_runner(qr)

        d_items_shuffled = shuffle_q.dequeue()
        for tensor, shape in zip(d_items_shuffled, shapes):
            tensor.set_shape(shape)
    else:
        d_items_shuffled = d_items

    padding_q = tf.PaddingFIFOQueue(
        capacity = (n_queue_threads+2)*batch_size,
        dtypes = dtypes,
        shapes = shapes)
    padding_op = padding_q.enqueue(d_items_shuffled)
    qr2 = tf.train.QueueRunner(padding_q, [padding_op]*n_queue_threads)
    tf.train.add_queue_runner(qr2)

    batched_data = padding_q.dequeue_many(batch_size)

    return dict(zip(d_keys, batched_data))

if __name__ == '__main__':
    training = build_input_machinery(["hv_training.tfrecords"], True,
        FLAGS.num_epochs, FLAGS.batch_size, FLAGS.min_after_dequeue,
        FLAGS.n_queue_threads)
    #validation = build_input_machinery(["hv_validation.tfrecords"], False,
    #    FLAGS.num_epochs, FLAGS.batch_size, FLAGS.min_after_dequeue,
    #    FLAGS.n_queue_threads)
    input_means = pu.load('means.pkl.gz')

    Model = getattr(model, FLAGS.model)
    train_step, loss =
        Model(num_units=FLAGS.hidden_units,
              num_layers=FLAGS.hidden_layers,
              inputs=training,
              input_means=input_means,
              training_keep_prob=FLAGS.dropout,
              learning_rate=FLAGS.learning_rate,
              batch_size=FLAGS.batch_size,
              layer_norm=FLAGS.layer_norm)
    train_op = m.train_step()


    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(max_to_keep=0)
    training_summary = tf.summary.scalar('training/loss', m.loss)
    #summary_op = tf.merge_all_summaries()

    sv = tf.train.Supervisor(is_chief=True,
                             logdir="./logs",
                             summary_op=training_summary,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)
    with sv.managed_session() as sess:
        while not sv.should_stop():
            sess.run(train_op)
        sv.stop()
