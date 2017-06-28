import tensorflow as tf
import pickle_utils as pu
import numpy as np
import itertools as it
import functools
import math
import os
from tqdm import tqdm

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer('batch_size', 64, 'batch size for training')
    flags.DEFINE_integer('num_samples', 16, 'samples to estimate expectation')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
    flags.DEFINE_integer('feature_i', None, 'Feature to learn')
    flags.DEFINE_string('layer_sizes', '[8]', 'layer sizes')
    flags.DEFINE_string('interpolation', None, 'Location of interpolation folder')
    flags.DEFINE_integer('patience', 20, 'Number of epochs to wait if'
                         'validation log-likelihood does not increase')
    del flags
    FLAGS = tf.app.flags.FLAGS

# Copyright 2017 Quim Llimona. https://pastebin.com/zM5c9xqX
def add_variable_scope(name=None, reuse=None):
    """Creates a variable_scope that contains all ops created by the function.
    The scope will default to the provided name or to the name of the function
    in CamelCase. If the function is a class constructor, it will default to
    the class name. It can also be specified with name='Name' at call time.
    """
    def _variable_scope_decorator(func):
        _name = name
        if _name is None:
            _name = func.__name__
            if _name == '__init__':
                _name = func.__class__.__name__
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            # Local, mutable copy of `name`.
            name_to_use = _name
            if 'name' in kwargs:
                if kwargs['name'] is not None:
                    name_to_use = kwargs['name']
                del kwargs['name']

            to_reuse = reuse
            if 'reuse' in kwargs:
                if kwargs['reuse'] is not None:
                    to_reuse = kwargs['reuse']
                del kwargs['reuse']

            with tf.variable_scope(name_to_use, reuse=to_reuse):
                return func(*args, **kwargs)
        return _wrapper
    return _variable_scope_decorator

@add_variable_scope(name="layer")
def layer(inputs, num_units, trainable, prior_variance=1.0, nlin=tf.nn.relu,
          initializer=tf.contrib.layers.variance_scaling_initializer):
    assert len(inputs.get_shape()) == 3

    param_shape = [int(inputs.get_shape()[2])+1, num_units]
    param_m = tf.get_variable("param_m", param_shape,
                              dtype=tf.float32,
                              trainable=trainable,
                              initializer=initializer())
    param_sigmoid_v = tf.get_variable("param_sigmoid_v", param_shape,
                              dtype=tf.float32,
                              trainable=trainable,
                              initializer=tf.constant_initializer(1.0))
    param_v = prior_variance*tf.sigmoid(param_sigmoid_v)

    num_samples = int(inputs.get_shape()[0])
    param_eps = tf.random_normal([num_samples] + param_shape, name="param_eps")

    param_sampled = tf.add(
        param_m, param_eps*tf.sqrt(param_v), name="param_sampled")
    param_sampled = tf.check_numerics(param_sampled, "param_sampled")

    out = tf.add(tf.matmul(inputs, param_sampled[:,:-1,:]), param_sampled[:,-1:,:],
                 name="activation")
    out = tf.check_numerics(out, "activation")
    if nlin is not None:
        out = nlin(out)
        out = tf.check_numerics(out, "out")

    if trainable:
        param_m_v = param_m/param_v

        log_f_Wb = [
            tf.reduce_sum(prior_variance*param_v/(prior_variance - param_v)
                        *tf.square(param_sampled), axis=[1,2]),
            tf.reduce_sum(param_m_v*param_sampled, axis=[1,2]),
        ]
        for tensor in log_f_Wb:
            f = tf.check_numerics(tf.expand_dims(tensor, axis=1), "f_W")
            f = tf.expand_dims(tensor, axis=1)
            tf.add_to_collection('f_W_components', f)

        log_Zq = [
            .5*tf.reduce_sum(tf.log((2*math.pi)*param_v)),
            tf.reduce_sum(param_m_v*param_m),
        ]
        for z in log_Zq:
            tf.add_to_collection('Z_q_components', z)

    return out

@add_variable_scope(name="log_likelihood")
def make_log_likelihood(preds, labels, n_outputs, log_noise, noise_var):
    diffs = labels-preds
    diffs = tf.where(tf.is_nan(diffs), tf.zeros_like(diffs), diffs)
    z_distances = diffs**2./noise_var
    # Marginal likelihood of the existing distances
    ll = -0.5*(n_outputs*math.log(2*math.pi) +
               # Ordinarily we would take `log |noise|`, where || is
               # determinant. Since `noise` is a diagonal matrix, the
               # determinant is just multiplying its entries. However we have
               # `log(noise)` instead, which means we can just take the sum of
               # it all.
               tf.reduce_sum(log_noise) + tf.reduce_sum(z_distances, axis=2))
    ll = tf.check_numerics(ll, "ll")
    return ll


@add_variable_scope(name=None)
def model(inputs, labels, N, num_samples, layer_sizes, alpha=0.5,
          trainable=True, include_samples=False, mean_X=None, mean_y=None, std_X=None, std_y=None):
    if mean_X is not None:
        inputs = (inputs - tf.constant(mean_X)) / tf.constant(std_X)
    if labels is None:
        n_outputs = len(mean_y)
    else:
        n_outputs = int(labels.get_shape()[-1])

    tiled_inputs = tf.tile(tf.expand_dims(inputs, axis=0), [num_samples, 1, 1])
    tiled_inputs = tf.check_numerics(tiled_inputs, "tiled_inputs")
    x = tiled_inputs
    for i, l in enumerate(layer_sizes):
        x = layer(x, l, trainable=trainable, nlin=tf.nn.tanh, name="layer_{:d}".format(i))
    x = layer(x, n_outputs, trainable=trainable, nlin=None, name="layer_out")
    preds = x
    del x, i, l

    log_noise = tf.get_variable("log_noise",
                                shape=[n_outputs],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1.),
                                trainable=trainable)
    results = {}
    noise_var = tf.exp(log_noise)
    noise_std = tf.sqrt(noise_var)
    if trainable:
        log_likelihood = make_log_likelihood(preds, labels, n_outputs,
                                             log_noise, noise_var)

        log_f_W = (1/N) * tf.add_n(tf.get_collection('f_W_components'))
        log_Zq = tf.add_n(tf.get_collection('Z_q_components'))

        # Eq 12 BB-alpha, 9-11 Depeweg
        energy = -log_Zq + (N/alpha)*(math.log(num_samples) -
            tf.reduce_mean(
                tf.reduce_logsumexp(
                    alpha*(log_likelihood - log_f_W)
                , axis=0)
            , axis=0))
        energy = tf.check_numerics(energy, "energy")
        assert energy.get_shape() == [], "Energy is a scalar"

        mean_prediction = tf.reduce_mean(preds, axis=0, name="mean_prediction")
        min_prediction = tf.subtract(tf.reduce_min(preds, axis=0), noise_std*2,
                                     name="min_prediction")
        max_prediction = tf.add(tf.reduce_max(preds, axis=0), noise_std*2,
                                name="max_prediction")

        results['energy'] = energy
        results['log_likelihood'] = tf.reduce_mean(log_likelihood, axis=0)
        results['mean_prediction'] = mean_prediction
        results['min_prediction'] = min_prediction
        results['max_prediction'] = max_prediction
        if mean_X is not None:
            results['mean_prediction'] = results['mean_prediction']*std_y + mean_y
            results['min_prediction'] = results['min_prediction']*std_y + mean_y
            results['max_prediction'] = results['max_prediction']*std_y + mean_y
    if include_samples:
        noise = tf.random_normal(
            tf.shape(preds), stddev=noise_std, name="noise")
        w_s = tf.reduce_mean(preds+noise, axis=0, name="white_samples")
        samples = tf.add(w_s*tf.constant(std_y), tf.constant(mean_y), name="samples")
        results['samples'] = samples
    return results

def build_sampler(inputs, feature_i, num_samples=8, layer_sizes=[64]):
    log_dir = os.path.join(FLAGS.interpolation,
                           'trained/num_{:d}'.format(feature_i))
    mX, sX, my, sy, N = pu.load(
        os.path.join(log_dir, 'means.pkl.gz'))
    m = model(inputs, None, N, num_samples, layer_sizes, trainable=False,
              include_samples=True,
              mean_X=mX, mean_y=my, std_X=sX, std_y=sy,
              name="num_{:d}".format(feature_i))
    validated_best = pu.load(os.path.join(log_dir, 'validated_best.pkl'))
    ckpt = os.path.join(log_dir, 'ckpt-{:d}'.format(validated_best))
    return m['samples'], ckpt

def build_input_machinery(dataset):
    X, y = map(lambda a: np.array(a, dtype=np.float32), pu.load(dataset))
    # We want a testing set of ~1000 examples
    test = np.random.random([len(X)]) < max(1000/len(X), 0.01)
    X_train = X[~test]
    y_train = y[~test,None]
    X_vali = X[test]
    y_vali = y[test,None]
    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    mean_y_train = np.mean(y_train, axis=0)
    std_y_train = np.std(y_train, axis=0)

    X_train = (X_train-mean_X_train)/std_X_train
    y_train = (y_train-mean_y_train)/std_y_train
    X_vali = (X_vali-mean_X_train)/std_X_train
    y_vali = (y_vali-mean_y_train)/std_y_train

    inputs = tf.placeholder(shape=[None]+list(X_train.shape[1:]), dtype=tf.float32)
    labels = tf.placeholder(shape=[None]+list(y_train.shape[1:]), dtype=tf.float32)

    X_test = (np.array(list(zip(*map(np.ravel, np.meshgrid(
        np.linspace(X[:,0].min(), X[:,0].max(), 20),
        np.linspace(X[:,1].min(), X[:,1].max(), 20))))), dtype=np.float32)
              - mean_X_train)/std_X_train

    return inputs, labels, X_train, y_train, X_vali, y_vali, X_test, \
        mean_X_train, std_X_train, mean_y_train, std_y_train, test


def main(_):
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    dataset = os.path.join(FLAGS.interpolation,
                           'num_{:d}.pkl.gz'.format(FLAGS.feature_i))
    log_dir = os.path.join(FLAGS.interpolation,
                           'trained/num_{:d}'.format(FLAGS.feature_i))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    ckpt_fname = os.path.join(log_dir, 'ckpt')

    inputs, labels, X_train, y_train, X_vali, y_vali, X_test, mean_X_train, \
        std_X_train, mean_y_train, std_y_train, test_mask = build_input_machinery(dataset)

    pu.dump((mean_X_train, std_X_train, mean_y_train, std_y_train,
             len(X_train)),
            os.path.join(log_dir, 'means.pkl.gz'))
    pu.dump(test_mask, os.path.join(log_dir, 'test_mask.pkl.gz'))

    global_step = tf.train.get_or_create_global_step()
    m = model(inputs, labels, len(X_train), FLAGS.num_samples,
              layer_sizes=eval(FLAGS.layer_sizes),
              name="num_{:d}".format(FLAGS.feature_i))
    train_op = (tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                .minimize(m['energy'], global_step=global_step))
    saver = tf.train.Saver(max_to_keep=0)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        slice_start = 0

        highest_likelihood = -np.inf
        times_waiting = 0
        try:
            for step in tqdm(range(0, 200000)):
                s = slice(slice_start, slice_start+FLAGS.batch_size)
                slice_start += FLAGS.batch_size
                if slice_start > len(X_train):
                    slice_start = 0

                if step%4000 == 0:
                    print("Doing step", step)
                    energy, ll, _ = sess.run([m['energy'], m['log_likelihood'], train_op], {
                        inputs: X_train[s], labels: y_train[s]})
                    train_ll = np.mean(ll)
                    #print("Training energy:", energy, "log-likelihood:", train_ll)
                    energy, ll, gs_val = sess.run([m['energy'], m['log_likelihood'], global_step], {
                        inputs: X_vali, labels: y_vali})
                    vali_ll = np.mean(ll)
                    #print("Validation energy:", energy, "log-likelihood:", vali_ll)
                    saver.save(sess, ckpt_fname, global_step=global_step)
                    if vali_ll > highest_likelihood:
                        times_waiting = 0
                        pu.dump(gs_val, os.path.join(log_dir, 'validated_best.pkl'))
                        highest_likelihood = vali_ll
                        print("Best validation likelihood so far:", vali_ll,
                              "(train", train_ll, ")")
                    else:
                        times_waiting += 1
                        if times_waiting >= FLAGS.patience:
                            break
                else:
                    sess.run(train_op, {
                        inputs: X_train[s], labels: y_train[s]})
        except KeyboardInterrupt:
            pass
        out_file = os.path.join(log_dir, 'test_prediction.pkl.gz')
        print("Writing prediction results to file:", out_file)
        pY, pY_min, pY_max = sess.run(
            [m['mean_prediction'], m['min_prediction'], m['max_prediction']],
            {inputs: X_test})
        pu.dump((pY, pY_min, pY_max), out_file)
        print(pY.shape, pY_min.shape, pY_max.shape)

if __name__ == '__main__':
    tf.app.run()
