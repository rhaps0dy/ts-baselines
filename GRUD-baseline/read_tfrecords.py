import tensorflow as tf
import pickle_utils as pu
import itertools as it

class FLAGS:
    num_epochs = 5064
    batch_size = 64
    min_after_dequeue = 5000
    n_queue_threads = 4

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
    return dict(map(
        lambda e: (e[0], tf.cast(e[1], convert[e[0]])) if e[0] in convert else e,
        it.chain(context.items(), sequence.items())))

def build_input_machinery(filenames):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=FLAGS.num_epochs)
    d = read_and_decode(filename_queue)
    d_keys, d_items = zip(*d.items())

    dtypes = list(a.dtype for a in d_items)
    shapes = list(a.get_shape() for a in d_items)

    shuffle_q = tf.RandomShuffleQueue(
        min_after_dequeue = FLAGS.min_after_dequeue,
        capacity = FLAGS.min_after_dequeue + (FLAGS.n_queue_threads+2)*FLAGS.batch_size,
        dtypes = dtypes)
    shuffle_op = shuffle_q.enqueue(d_items)
    qr = tf.train.QueueRunner(shuffle_q, [shuffle_op]*FLAGS.n_queue_threads)
    tf.train.add_queue_runner(qr)

    d_items_shuffled = shuffle_q.dequeue()
    for tensor, shape in zip(d_items_shuffled, shapes):
        tensor.set_shape(shape)

    padding_q = tf.PaddingFIFOQueue(
        capacity = (FLAGS.n_queue_threads+2)*FLAGS.batch_size,
        dtypes = dtypes,
        shapes = shapes)
    padding_op = padding_q.enqueue(d_items_shuffled)
    qr2 = tf.train.QueueRunner(padding_q, [padding_op]*FLAGS.n_queue_threads)
    tf.train.add_queue_runner(qr2)

    batched_data = padding_q.dequeue_many(FLAGS.batch_size)

    return dict(zip(d_keys, batched_data))

if __name__ == '__main__':
    batch = build_input_machinery(["hour_ventilation.tfrecords"])

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                print(sess.run([batch['numerical_ts_dt']]))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
