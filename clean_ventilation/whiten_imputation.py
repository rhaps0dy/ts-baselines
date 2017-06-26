import tensorflow as tf
import pickle_utils as pu
import itertools as it
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '..', 'GRUD-baseline'))
from imputation_read_tfrecords import build_input_machinery

flags = tf.app.flags
flags.DEFINE_string('dataset', '', 'The file to use as input dataset')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_string('command', None, '[Means, Stddevs]')
del flags
FLAGS = tf.app.flags.FLAGS

class Means:
    def __init__(self, dataset, feature_numbers):
        self.total_n = 0
        self.values = {}
        self.counts = {}
        for k in dataset.keys():
            self.values[k] = np.zeros([feature_numbers['numerical_ts']], dtype=np.float)
            self.counts[k] = np.zeros([feature_numbers['numerical_ts']], dtype=np.int)

    def process(self, data):
        n_processed = data['num_ts'].shape[0]
        self.total_n += n_processed
        for k in 'num_ts', 'num_forward':
            self.values[k] += np.sum(data[k], axis=0)
        self.values['num_labels'] += np.nansum(data['num_labels'], axis=0)
        self.counts['num_labels'] += np.sum(~np.isnan(data['num_labels']), axis=0)
        return n_processed

    def finish(self):
        self.counts['num_ts'][...] = self.total_n
        self.counts['num_forward'][...] = self.total_n
        pu.dump({'counts': self.counts,
                 'values': self.values,
                 'total_n': self.total_n},
                os.path.join(os.path.dirname(FLAGS.dataset),
                             'whiten_imputation.pkl.gz'))
class Stddevs:
    def __init__(self, dataset, feature_numbers):
        self.fpath = os.path.join(os.path.dirname(FLAGS.dataset),
                             'whiten_imputation.pkl.gz')
        self.save = pu.load(self.fpath)
        self.means = {}
        self.stddevs = {}
        self.stddevs_mul = {}
        for k in self.save['counts'].keys():
            self.means[k] = self.save['values'][k] / self.save['counts'][k]
            self.stddevs[k] = np.zeros_like(self.means[k])
            self.means[k] = self.means[k][None,:]
            self.stddevs_mul[k] = 1/(self.save['counts'][k]-1)

    def process(self, data):
        n_processed = data['num_ts'].shape[0]
        for k in 'num_ts', 'num_forward':
            self.stddevs[k] += self.stddevs_mul[k] * np.sum((self.means[k] - data[k])**2, axis=0)
        k = 'num_labels'
        self.stddevs[k] += self.stddevs_mul[k] * np.nansum((self.means[k] - data[k])**2, axis=0)
        return n_processed

    def finish(self):
        self.save['stddevs'] = self.stddevs
        pu.dump(self.save, self.fpath)

def main(_):
    assert os.path.isfile(FLAGS.dataset)

    feature_numbers = pu.load(os.path.join(
        os.path.dirname(FLAGS.dataset), 'feature_numbers.pkl.gz'))

    dataset = build_input_machinery([FLAGS.dataset], feature_numbers, False, 1,
                                    FLAGS.batch_size, None, 1)
    keys = list(dataset.keys())
    Klass = {'Means': Means, 'Stddevs': Stddevs}[FLAGS.command]
    klass = Klass(dataset, feature_numbers)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        queue_threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        progress_bar = tqdm()
        try:
            while not coord.should_stop():
                data = dict(zip(keys, sess.run(list(map(dataset.__getitem__, keys)))))
                progress_bar.update(n=klass.process(data))
        except tf.errors.OutOfRangeError:
            pass
        klass.finish()

        coord.request_stop()
        coord.join(queue_threads)


if __name__ == '__main__':
    tf.app.run()
