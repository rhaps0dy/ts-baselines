import tensorflow as tf
import pickle_utils as pu
import itertools as it
import numpy as np
import collections
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '..', 'GRUD-baseline'))
from read_tfrecords import build_input_machinery

flags = tf.app.flags
flags.DEFINE_string('dataset', '', 'The file to use as input dataset')
del flags
FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 64

def main(_):
    assert os.path.exists(FLAGS.dataset)

    if not os.path.exists('interpolation'):
        os.mkdir('interpolation')
    print("Will write files to ./interpolation")

    feature_numbers = pu.load(os.path.join(
        os.path.dirname(FLAGS.dataset), 'feature_numbers.pkl.gz'))
    dataset = build_input_machinery([FLAGS.dataset], feature_numbers, False, 1,
                                    BATCH_SIZE, None, 1)

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

    category_counts = list(collections.Counter() for _ in range(cat_shape[1]))
    numerical_averages = np.zeros(num_shape[1], dtype=np.float32)
    numerical_averages_counts = np.zeros(num_shape[1], dtype=np.int)
    numerical_averages_counts_2 = np.zeros(num_shape[1], dtype=np.int)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        progress_bar = tqdm()
        try:
            while not coord.should_stop():
                numerical_ts, categorical_ts, length = sess.run(list(map(
                    dataset.__getitem__, ['numerical_ts', 'categorical_ts', 'length'])))
                for feature_i in range(cat_shape[1]):
                    for j, l in enumerate(length):
                        category_counts[feature_i].update(categorical_ts[j,:l,feature_i].flatten())

                progress_bar.update(n=numerical_ts.shape[0])
                if numerical_ts.shape[0] < num_shape[0]:
                    # Pad so that the first dimension is always BATCH_SIZE
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
                        numerical_averages[f] += num_ts[ex,f]
                        numerical_averages_counts_2[f] += 1
                    numerical_averages_counts += np.sum(num_is_valid, axis=0)

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
        except tf.errors.OutOfRangeError:
            pass

        #for i, niX, niY in zip(it.count(0), num_interpolate_X, num_interpolate_y):
        #    pu.dump((niX, niY), 'interpolation/num_{:d}.pkl.gz'.format(i))
        #for i, ciX, ciY in zip(it.count(0), cat_interpolate_X, cat_interpolate_y):
        #    pu.dump((ciX, ciY), 'interpolation/cat_{:d}.pkl.gz'.format(i))
        nums_cats = pu.load('dataset/number_of_categories_200.pkl.gz')['categorical_ts']
        for i, (count, n_cats) in enumerate(zip(category_counts, nums_cats)):
            a = np.zeros(n_cats, dtype=np.int)
            print("Category", i, "counts", count)
            for j in range(n_cats):
                if j in count:
                    a[j] = count[j]
            pu.dump(a, 'interpolation/counts_cat_{:d}.pkl.gz'.format(i))
        pu.dump((numerical_averages, numerical_averages_counts_2), 'dataset/means.pkl.gz')
        assert np.all(numerical_averages_counts_2 == numerical_averages_counts)

        coord.request_stop()
        coord.join(queue_threads)

if __name__ == '__main__':
    tf.app.run()
