import tensorflow as tf
import pickle_utils as pu
import numpy as np
import itertools as it
import functools
import math

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64, 'batch size for training')
flags.DEFINE_integer('num_samples', 20, 'samples to estimate expectation')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for ADAM')
flags.DEFINE_string('dataset', '', 'dataset to load')
flags.DEFINE_string('log_dir', 'test_bb_alpha', 'directory to log to')
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
def layer(inputs, num_units, prior_variance=1.0, nlin=tf.nn.relu,
          initializer=tf.contrib.layers.variance_scaling_initializer):
    assert len(inputs.get_shape()) == 3

    param_shape = [int(inputs.get_shape()[2])+1, num_units]
    param_m = tf.get_variable("param_m", param_shape,
                              dtype=tf.float32,
                              trainable=True,
                              initializer=initializer())
    param_sigmoid_v = tf.get_variable("param_sigmoid_v", param_shape,
                              dtype=tf.float32,
                              trainable=True,
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

    param_m_v = param_m/param_v

    log_f_Wb = [
        tf.reduce_sum(prior_variance*param_v/(prior_variance - param_v)
                      *tf.square(param_sampled), axis=[1,2]),
        tf.reduce_sum(param_m_v*param_sampled, axis=[1,2]),
    ]
    for tensor in log_f_Wb:
        f = tf.check_numerics(tf.expand_dims(tensor, axis=1), "f_W")
        tf.add_to_collection('f_W_components', f)

    log_Zq = [
        .5*tf.reduce_sum(tf.log((2*math.pi)*param_v)),
        tf.reduce_sum(param_m_v*param_m),
    ]
    for z in log_Zq:
        tf.add_to_collection('Z_q_components', z)

    return out

@add_variable_scope(name="log_likelihood")
def make_log_likelihood(preds, labels, log_noise):
    noise = tf.exp(log_noise)
    ll = -0.5*(int(labels.get_shape()[-1])*math.log(2*math.pi) +
               # Ordinarily we would take `log |noise|`, where || is
               # determinant. Since `noise` is a diagonal matrix, the
               # determinant is just multiplying its entries. However we have
               # `log(noise)` instead, which means we can just take the sum of
               # it all.
               tf.reduce_sum(log_noise) + tf.reduce_sum(
                   (labels-preds)**2./noise, axis=2))
    return ll, noise


@add_variable_scope(name="model")
def model(inputs, labels, N, num_samples, alpha=0.5):
    tiled_inputs = tf.tile(tf.expand_dims(inputs, axis=0), [num_samples, 1, 1])
    tiled_inputs = tf.check_numerics(tiled_inputs, "tiled_inputs")
    l1 = layer(tiled_inputs, 16, nlin=tf.nn.relu, name="layer_1")
    l1 = tf.check_numerics(l1, "l1")
    l2 = layer(l1, 1, nlin=None, name="layer_2")
    l2 = tf.check_numerics(l2, "l2")
    preds = l2

    log_noise = tf.get_variable("log_noise",
                                shape=[labels.get_shape()[-1]],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1.),
                                trainable=True)
    log_likelihood, var = make_log_likelihood(preds, labels, log_noise)

    log_f_W = (1/N) * tf.add_n(tf.get_collection('f_W_components'))
    log_Zq = tf.add_n(tf.get_collection('Z_q_components'))

    # Eq 12 BB-alpha, 9-11 Depeweg
    energy = -log_Zq + (N/alpha)*(math.log(num_samples) -
        tf.reduce_mean(
            tf.reduce_logsumexp(
                alpha*(log_likelihood - log_f_W)
            , axis=0)
        , axis=0))
    assert energy.get_shape() == [], "Energy is a scalar"

    mean_prediction = tf.reduce_mean(preds, axis=0, name="mean_prediction")
    stddev = tf.sqrt(var)
    min_prediction = tf.subtract(tf.reduce_min(preds, axis=0), stddev*2, name="min_prediction")
    max_prediction = tf.add(tf.reduce_max(preds, axis=0), stddev*2, name="max_prediction")

    return {'energy': energy,
            'log_likelihood': tf.reduce_mean(log_likelihood, axis=0),
            'mean_prediction': mean_prediction,
            'min_prediction': min_prediction,
            'max_prediction': max_prediction,
            }

def build_input_machinery(dataset):
    X, y = map(lambda a: np.array(a, dtype=np.float32), pu.load(dataset))
    test = np.random.random([len(X)]) < 0.01
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

    return inputs, labels, X_train, y_train, X_vali, y_vali, X_test, mean_y_train, std_y_train


def main(_):
    inputs, labels, X_train, y_train, X_vali, y_vali, X_test, mean_y_train, std_y_train = (
        build_input_machinery(FLAGS.dataset))

    global_step = tf.train.get_or_create_global_step()
    m = model(inputs, labels, len(X_train), FLAGS.num_samples)
    train_op = (tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
                .minimize(m['energy'], global_step=global_step))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        slice_start = 0
        try:
            for step in it.count(0):
                s = slice(slice_start, slice_start+FLAGS.batch_size)
                slice_start += FLAGS.batch_size
                if slice_start > len(X_train):
                    slice_start = 0

                if step%2000 == 0:
                    print("Doing step", step)
                    energy, ll, _ = sess.run([m['energy'], m['log_likelihood'], train_op], {
                        inputs: X_train[s], labels: y_train[s]})
                    print("Training energy:", energy, "log-likelihood:", np.mean(ll))
                    energy, ll = sess.run([m['energy'], m['log_likelihood']], {
                        inputs: X_vali, labels: y_vali})
                    print("Validation energy:", energy, "log-likelihood:", np.mean(ll))
                else:
                    sess.run(train_op, {
                        inputs: X_train[s], labels: y_train[s]})
        except KeyboardInterrupt:
            pass
        print("Writing prediction results to file: 'out.pkl.gz'")
        pY, pY_min, pY_max = sess.run(
            [m['mean_prediction'], m['min_prediction'], m['max_prediction']],
            {inputs: X_test})
        pY = pY*std_y_train + mean_y_train
        pY_min = pY_min*std_y_train + mean_y_train
        pY_max = pY_max*std_y_train + mean_y_train
        pu.dump((pY, pY_min, pY_max), 'out.pkl.gz')
        print(pY.shape, pY_min.shape, pY_max.shape)

if __name__ == '__main__':
    tf.app.run()
