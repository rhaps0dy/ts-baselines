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
flags.DEFINE_string('command', 'GatherInterpolationInfo', 'The command to perform')
del flags
FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 64

class GatherInterpolationInfo:
    keys = ['numerical_ts', 'categorical_ts', 'length']
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_shape = [BATCH_SIZE, int(self.dataset['numerical_ts'].get_shape()[2])]
        self.last_num_ts = np.empty(self.num_shape, dtype=np.float32)
        self.num_ts_dt = np.zeros(self.num_shape, dtype=np.int)

        self.cat_shape = [BATCH_SIZE, int(self.dataset['categorical_ts'].get_shape()[2])]
        self.last_cat_ts = -np.ones(self.cat_shape, dtype=np.int32) # all -1
        self.cat_ts_dt = np.zeros(self.cat_shape, dtype=np.int)

        self.num_interpolate_X = list([] for _ in range(self.num_shape[1]))
        self.num_interpolate_y = list([] for _ in range(self.num_shape[1]))
        self.cat_interpolate_X = list([] for _ in range(self.cat_shape[1]))
        self.cat_interpolate_y = list([] for _ in range(self.cat_shape[1]))

        self.category_counts = list(collections.Counter()
                                    for _ in range(self.cat_shape[1]))
        self.numerical_averages = np.zeros(self.num_shape[1], dtype=np.float32)
        self.numerical_averages_counts = np.zeros(self.num_shape[1],
                                                  dtype=np.int)
        self.numerical_averages_counts_2 = np.zeros(self.num_shape[1],
                                                    dtype=np.int)

    def process(self, data):
        numerical_ts, categorical_ts, length = map(data.__getitem__,
                                                   self.keys)
        for feature_i in range(self.cat_shape[1]):
            for j, l in enumerate(length):
                self.category_counts[feature_i].update(
                    categorical_ts[j,:l,feature_i].flatten())

        if numerical_ts.shape[0] < self.num_shape[0]:
            # Pad so that the first dimension is always BATCH_SIZE
            numerical_ts = np.pad(
                numerical_ts,
                [(0, self.num_shape[0]-numerical_ts.shape[0]), (0, 0), (0, 0)],
                'constant', constant_values=np.nan)
            categorical_ts = np.pad(
                categorical_ts,
                [(0, self.cat_shape[0]-categorical_ts.shape[0]), (0, 0), (0, 0)],
                'constant', constant_values=-1)
            length = np.pad(
                length,
                [(0, self.num_shape[0]-length.shape[0])],
                'constant', constant_values=0)

        self.last_num_ts[:] = np.nan
        self.num_ts_dt[:] = 0
        self.last_cat_ts[:] = -1
        self.cat_ts_dt[:] = 0
        for t in range(numerical_ts.shape[1]):
            num_ts = numerical_ts[:,t,:]
            cat_ts = categorical_ts[:,t,:]
            assert num_ts.shape[1] == self.num_shape[1]
            assert cat_ts.shape[1] == self.cat_shape[1]

            is_in_range = (t < length)[:,None]
            num_is_valid = is_in_range * (~np.isnan(num_ts))

            for ex, f in zip(*num_is_valid.nonzero()):
                n = self.last_num_ts[ex,f]
                if not np.isnan(n):
                    self.num_interpolate_X[f].append((n, self.num_ts_dt[ex,f]))
                    self.num_interpolate_y[f].append((num_ts[ex,f]))
                self.numerical_averages[f] += num_ts[ex,f]
                self.numerical_averages_counts_2[f] += 1
            self.numerical_averages_counts += np.sum(num_is_valid, axis=0)

            cat_is_valid = is_in_range * (cat_ts >= 0)
            for ex, f in zip(*cat_is_valid.nonzero()):
                n = self.last_cat_ts[ex,f]
                if n >= 0:
                    self.cat_interpolate_X[f].append((n, self.cat_ts_dt[ex,f]))
                    self.cat_interpolate_y[f].append((cat_ts[ex,f]))

            self.num_ts_dt += 1
            self.num_ts_dt[num_is_valid] = 0
            self.last_num_ts[num_is_valid] = num_ts[num_is_valid]
            self.cat_ts_dt += 1
            self.cat_ts_dt[cat_is_valid] = 0
            self.last_cat_ts[cat_is_valid] = cat_ts[cat_is_valid]

    def finish(self):
        for i, niX, niY in zip(it.count(0), self.num_interpolate_X, self.num_interpolate_y):
            pu.dump((niX, niY), 'interpolation/num_{:d}.pkl.gz'.format(i))
        for i, ciX, ciY in zip(it.count(0), self.cat_interpolate_X, self.cat_interpolate_y):
            pu.dump((ciX, ciY), 'interpolation/cat_{:d}.pkl.gz'.format(i))
        nums_cats = pu.load('dataset/number_of_categories_200.pkl.gz')['categorical_ts']
        for i, (count, n_cats) in enumerate(zip(self.category_counts, nums_cats)):
            a = np.zeros(n_cats, dtype=np.int)
            print("Category", i, "counts", count)
            for j in range(n_cats):
                if j in count:
                    a[j] = count[j]
            pu.dump(a, 'interpolation/counts_cat_{:d}.pkl.gz'.format(i))
        pu.dump((self.numerical_averages, self.numerical_averages_counts_2), 'dataset/means.pkl.gz')
        assert np.all(self.numerical_averages_counts_2 == self.numerical_averages_counts)

class AddInterpolationInputs:
    keys = ['icustay_id', 'numerical_static', 'categorical_static',
                'numerical_ts_dt', 'categorical_ts_dt', 'time_until_label',
                'label', 'numerical_ts', 'categorical_ts', 'treatments_ts',
                'ventilation_ends']

    def __init__(self, dataset):
        self.dataset = dataset
        self.writer = tf.python_io.TFRecordWriter(FLAGS.dataset+'-new')
        self.writer.__enter__()

    @staticmethod
    def build_dt(d, d_valid, initial_dt):
        d = np.transpose(d, [1,0,2])
        d_valid = np.transpose(d_valid, [1,0,2])
        dt = np.zeros_like(d, dtype=np.int64)
        dt[0,:] = initial_dt
        for t in range(1, dt.shape[0]):
            dt[t] = dt[t-1]+1
            dt[t,d_valid[t]] = 0
        return np.transpose(dt, [1,0,2])

    @staticmethod
    def build_impute_forward(d, d_valid, means):
        d = np.transpose(d, [1,0,2])
        d_valid = np.transpose(d_valid, [1,0,2])
        impute = np.copy(d)
        initial_valid = d_valid[0,:]
        impute[0,initial_valid] = means[0,initial_valid]
        for t in range(1, impute.shape[0]):
            to_carry_forward = ~d_valid[t,:]
            impute[t,to_carry_forward] = impute[t-1,to_carry_forward]
        return np.transpose(impute, [1,0,2])

    def process(self, data):
        is_in_range = (t < data['length'])[:,None]
        num_is_valid = is_in_range * (~np.isnan(data['numerical_ts']))
        data['numerical_ts_forward'] = (
            build_dt(data['numerical_ts'], is_valid, data['numerical_ts_dt']))

        cat_is_valid = is_in_range * (data['categorical_ts'] >= 0)

    def finish(self):
        self.writer.__exit__(None, None, None)


def main(_):
    assert os.path.exists(FLAGS.dataset)

    if not os.path.exists('interpolation'):
        os.mkdir('interpolation')
    print("Will write files to ./interpolation")

    feature_numbers = pu.load(os.path.join(
        os.path.dirname(FLAGS.dataset), 'feature_numbers.pkl.gz'))
    dataset = build_input_machinery([FLAGS.dataset], feature_numbers, False, 1,
                                    BATCH_SIZE, None, 1)

    Klass = getattr(globals(), FLAGS.command)
    klass = Klass()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        progress_bar = tqdm()
        try:
            while not coord.should_stop():
                data = dict(zip(klass.keys,
                                sess.run(list(map(dataset.__getitem__, klass.keys)))))
                n_processed = data['numerical_ts'].shape[0]
                klass.process(data)
                progress_bar.update(n=n_processed)
        except tf.errors.OutOfRangeError:
            pass

        klass.finish()

        coord.request_stop()
        coord.join(queue_threads)

if __name__ == '__main__':
    tf.app.run()

import unittest
class TestAddInterpolation(unittest.TestCase):
    def test_build_dt(self):
        a = np.array([[[0, 1],
                       [0, 0],
                       [1, 0]],
                      [[1, 1],
                       [1, 0],
                       [0, 0]]])
        self.assertTrue(np.all(AddInterpolationInputs.build_dt(a, a==0) ==
                         np.array([[[0, 1000],
                                    [0, 0],
                                    [1, 0]],
                                   [[1000, 1000],
                                    [1001, 0],
                                    [0, 0]]])))


