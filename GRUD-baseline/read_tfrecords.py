import tensorflow as tf
import pickle_utils as pu
import itertools as it
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '..', 'clean_ventilation'))
import common

def read_and_decode(filename_queue, f_num, keys=None):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        'icustay_id': tf.FixedLenFeature([], tf.int64),
        'numerical_static': tf.FixedLenFeature(
            [f_num['numerical_static']], dtype=tf.float32),
        'categorical_static': tf.FixedLenFeature(
            [f_num['categorical_static']], dtype=tf.int64),
        'numerical_ts_dt': tf.FixedLenFeature(
            [f_num['numerical_ts']], dtype=tf.float32),
        'categorical_ts_dt': tf.FixedLenFeature(
            [f_num['categorical_ts']], dtype=tf.float32),
    }
    if keys is not None:
        context_features = dict(filter(lambda t: t[0] in keys,
                                       context_features.items()))
    sequence_features = {
        'time_until_label': tf.FixedLenSequenceFeature([], tf.float32),
        'label': tf.FixedLenSequenceFeature([], tf.float32),
        'ventilation_ends': tf.FixedLenSequenceFeature([], tf.int64),
    }
    for name, typ in common.EXAMPLE['sequence'].items():
        str_ts = '_ts'
        i = name.find(str_ts)
        if i >= 0:
            typ = ('float32' if typ == 'float' else typ)
            sequence_features[name] = tf.FixedLenSequenceFeature(
                [f_num[name[:i+len(str_ts)]]], dtype=getattr(tf, typ))
    if keys is not None:
        sequence_features = dict(filter(lambda t: t[0] in keys,
                                        sequence_features.items()))

    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features)
    convert = {'icustay_id': tf.int32, 'categorical_static': tf.int32,
               'categorical_ts': tf.int32, 'ventilation_ends': tf.int32}

    if keys is not None:
        assert 'numerical_ts' in keys
    additional_items = [('length', tf.shape(sequence['numerical_ts'])[0])]
    if keys is None or 'n_ventilations' in keys:
        additional_items.append(
            ('n_ventilations', tf.shape(sequence['ventilation_ends'])[0]))

    return dict(map(
        lambda e: (e[0], tf.cast(e[1], convert[e[0]])) if e[0] in convert else e,
        it.chain(context.items(), sequence.items(), additional_items)))

def build_input_machinery(filenames, feature_numbers, do_shuffle, num_epochs,
                          batch_size, min_after_dequeue, n_queue_threads, keys=None):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=num_epochs)
    d = read_and_decode(filename_queue, feature_numbers, keys=keys)
    d_keys, d_items = zip(*d.items())

    dtypes = list(a.dtype for a in d_items)
    shapes = list(a.get_shape() for a in d_items)

    if do_shuffle:
        shuffle_q = tf.RandomShuffleQueue(
            min_after_dequeue=min_after_dequeue,
            capacity=(min_after_dequeue +
                      (n_queue_threads + 2) * batch_size),
            dtypes=dtypes)
        shuffle_op = shuffle_q.enqueue(d_items)
        qr = tf.train.QueueRunner(shuffle_q, [shuffle_op] * n_queue_threads)
        tf.train.add_queue_runner(qr)

        d_items_shuffled = shuffle_q.dequeue()
        for tensor, shape in zip(d_items_shuffled, shapes):
            tensor.set_shape(shape)
    else:
        d_items_shuffled = d_items

    padding_q = tf.PaddingFIFOQueue(
        capacity=(n_queue_threads + 2) * batch_size,
        dtypes=dtypes,
        shapes=shapes)
    padding_op = padding_q.enqueue(d_items_shuffled)
    qr2 = tf.train.QueueRunner(padding_q, [padding_op] * n_queue_threads)
    tf.train.add_queue_runner(qr2)

    batched_data = padding_q.dequeue_up_to(batch_size)

    return dict(zip(d_keys, batched_data))
