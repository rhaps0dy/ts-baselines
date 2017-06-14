import tensorflow as tf
import pickle_utils as pu
import itertools as it
import numpy as np
import os.path

from read_tfrecords import build_input_machinery

flags = tf.app.flags
flags.DEFINE_string('dataset', '', 'The file to use as input dataset')
del flags
FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 64

def main(_):
    assert os.path.exists(FLAGS.dataset)
    dataset = build_input_machinery([FLAGS.dataset], False, 1, BATCH_SIZE, None, 1)

    num_shape = [BATCH_SIZE, int(dataset['numerical_ts'].get_shape()[2])]
    last_num_ts = np.empty(num_shape, dtype=np.float32)
    num_ts_dt = np.zeros(num_shape, dtype=np.int)

    cat_shape = [BATCH_SIZE, int(dataset['categorical_ts'].get_shape()[2])]
    last_cat_ts = -np.ones(cat_shape, dtype=np.int32) # all -1
    cat_ts_dt = np.zeros(cat_shape, dtype=np.int)

    num_interpolate_X = list([] for _ in range(num_shape[1]))
    num_interpolate_y = list([] for _ in range(num_shape[1]))
    cat_interpolate_X = list([] for _ in range(cat_shape[1]))
    cat_interpolate_y = list([] for _ in range(cat_shape[1]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            n_processed = 0
            while not coord.should_stop():
                numerical_ts, categorical_ts, length = sess.run(list(map(
                    dataset.__getitem__, ['numerical_ts', 'categorical_ts', 'length'])))
                n_processed += numerical_ts.shape[0]
                if numerical_ts.shape[0] < num_shape[0]:
                    numerical_ts = np.pad(
                        numerical_ts,
                        [(0, num_shape[0]-numerical_ts.shape[0]), (0, 0), (0, 0)],
                        'constant', constant_values=np.nan)
                    categorical_ts = np.pad(
                        categorical_ts,
                        [(0, cat_shape[0]-categorical_ts.shape[0]), (0, 0), (0, 0)],
                        'constant', constant_values=-1)
                    length = np.pad(
                        length,
                        [(0, num_shape[0]-length.shape[0])],
                        'constant', constant_values=0)
                last_num_ts[:] = np.nan
                num_ts_dt[:] = 0
                last_cat_ts[:] = -1
                cat_ts_dt[:] = 0
                for t in range(numerical_ts.shape[1]):
                    num_ts = numerical_ts[:,t,:]
                    cat_ts = categorical_ts[:,t,:]
                    assert num_ts.shape[1] == num_shape[1]
                    assert cat_ts.shape[1] == cat_shape[1]

                    is_in_range = (t < length)[:,None]
                    num_is_valid = is_in_range * (~np.isnan(num_ts))

                    for ex, f in zip(*num_is_valid.nonzero()):
                        n = last_num_ts[ex,f]
                        if not np.isnan(n):
                            num_interpolate_X[f].append((n, num_ts_dt[ex,f]))
                            num_interpolate_y[f].append((num_ts[ex,f]))

                    cat_is_valid = is_in_range * (cat_ts >= 0)
                    for ex, f in zip(*cat_is_valid.nonzero()):
                        n = last_cat_ts[ex,f]
                        if n >= 0:
                            cat_interpolate_X[f].append((n, cat_ts_dt[ex,f]))
                            cat_interpolate_y[f].append((cat_ts[ex,f]))

                    num_ts_dt += 1
                    num_ts_dt[num_is_valid] = 0
                    last_num_ts[num_is_valid] = num_ts[num_is_valid]
                    cat_ts_dt += 1
                    cat_ts_dt[cat_is_valid] = 0
                    last_cat_ts[cat_is_valid] = cat_ts[cat_is_valid]
                print("Processed {:d} examples".format(n_processed))
        except tf.errors.OutOfRangeError:
            pass

        for i, niX, niY in zip(it.count(0), num_interpolate_X, num_interpolate_y):
            pu.dump((niX, niY), 'interpolation/num_{:d}.pkl.gz'.format(i))
        for i, ciX, ciY in zip(it.count(0), cat_interpolate_X, cat_interpolate_y):
            pu.dump((ciX, ciY), 'interpolation/cat_{:d}.pkl.gz'.format(i))

        coord.request_stop()
        coord.join(queue_threads)

if __name__ == '__main__':
    tf.app.run()
