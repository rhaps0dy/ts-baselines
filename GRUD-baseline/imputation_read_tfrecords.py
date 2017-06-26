import tensorflow as tf
import pickle_utils as pu
import itertools as it

def build_input_machinery(filenames, feature_numbers, do_shuffle, num_epochs,
                          batch_size, min_after_dequeue, n_queue_threads, keys=None):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    if do_shuffle:
        serialized = tf.train.shuffle_batch(
            tensors=[serialized_example],
            batch_size=batch_size,
            capacity=(min_after_dequeue +
                      (n_queue_threads + 2) * batch_size),
            min_after_dequeue=min_after_dequeue,
            num_threads=n_queue_threads)
    else:
        serialized = tf.train.batch(
            tensors=[serialized_example],
            batch_size=batch_size,
            capacity=(n_queue_threads + 2) * batch_size,
            num_threads=n_queue_threads)

    shape = [feature_numbers['numerical_ts']]
    features = {
        'num_ts': tf.FixedLenFeature(shape, dtype=tf.float32),
        'num_forward': tf.FixedLenFeature(shape, dtype=tf.float32),
        'num_labels': tf.FixedLenFeature(shape, dtype=tf.float32),
    }
    if keys is not None:
        features = dict(filter(lambda t: t[0] in keys,
                                       features.items()))
    return tf.parse_example(serialized, features=features)
