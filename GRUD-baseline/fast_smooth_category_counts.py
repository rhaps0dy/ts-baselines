import pickle_utils as pu
import numpy as np
import os
import tensorflow as tf

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer('feature_i', None, 'Feature to learn')
    flags.DEFINE_integer('patience', 5, 'Number of epochs to wait if'
                         'validation log-likelihood does not increase')
    flags.DEFINE_integer('num_set_draws', 30,
                         'Number of random training-validation splits')
    flags.DEFINE_string('interpolation', None, 'Location of interpolation folder')
    del flags
    FLAGS = tf.app.flags.FLAGS

def make_training_test(counts, proportion):
    counts_ = np.reshape(counts, [-1])
    total_examples = np.sum(counts_)
    counts_ = counts_ / total_examples

    test_set = np.random.multinomial(int(total_examples*proportion),
                                        pvals=counts_)
    test_set = np.reshape(test_set, counts.shape)
    training_set = counts.astype(np.int) - test_set
    test_set += np.clip(training_set, -1000000000, 0)
    training_set = counts.astype(np.int) - test_set
    assert np.all(training_set >= 0)
    return training_set, test_set

def training_model(training_set, test_set, total_counts, len_t, N_cats):
    # Prepare compact training and test sets
    np_filter = np.zeros([len_t*2+1], dtype=np.float32)
    np_filter[:len_t] = np.arange(len_t, 0, -1)
    np_filter[len_t+1:] = np.arange(len_t)+1

    training_time_filter = (np.sum(np.sum(training_set, axis=0), axis=0) != 0)
    test_time_filter = (np.sum(np.sum(test_set, axis=0), axis=0) != 0)

    compact_training = training_set[:,:,training_time_filter]
    assert np.sum(training_set[:,:,~training_time_filter]) == 0.0
    filter_matrix = np.zeros([np.sum(test_time_filter),
                              compact_training.shape[2]], dtype=np.float32)
    for i, t in enumerate(test_time_filter.nonzero()[0]):
        filter_matrix[i,:] = np_filter[len_t-t:len_t*2-t][training_time_filter]

    np_smoothing = np.reshape(total_counts, [1,1,N_cats])/np.sum(total_counts)

    filter_matrix = -filter_matrix
    compact_training = np.transpose(compact_training, [0,2,1])
    compact_test = np.transpose(test_set[:,:,test_time_filter], [0,2,1])

    # Take only the categories that have at least 1 example
    cat_f = (np_smoothing.flatten() > 0)
    cat_f_mult = np.tile(cat_f, [FLAGS.num_set_draws])
    unzero_training = compact_training.astype(np.float32)[cat_f_mult,:,:][:,:,cat_f]
    unzero_test = compact_test.astype(np.float32)[cat_f_mult,:][:,:,cat_f]
    smoothing = np_smoothing.astype(np.float32)[:,:,cat_f]

    # Create tensorflow variables and operations
    scale = tf.get_variable("scale", shape=[], dtype=tf.float32,
                            trainable=True,
                            initializer=tf.constant_initializer(1))
    m1 = tf.exp(filter_matrix*scale)
    m1 = tf.tile(tf.expand_dims(m1, 0), [unzero_training.shape[0], 1, 1])
    print(m1.get_shape())
    print(unzero_training.shape)
    n = tf.matmul(m1, unzero_training) + smoothing
    print(n.get_shape())
    p = n/tf.reduce_sum(n, axis=2, keep_dims=True)
    print(p.get_shape())
    print(unzero_test.shape)
    tf_ll = tf.reduce_sum(unzero_test*tf.log(p))
    return tf_ll, scale

def main(_):
    print("Doing category", FLAGS.feature_i)
    log_dir = os.path.join(FLAGS.interpolation,
                           'trained/cat_{:d}'.format(FLAGS.feature_i))
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    scale_fname = os.path.join(log_dir, 'scale.pkl')

    X, y = pu.load(os.path.join(
        FLAGS.interpolation, 'cat_{:d}.pkl.gz'.format(FLAGS.feature_i)))
    # The last item in `total_counts` is the number of missing
    total_counts = pu.load(os.path.join(
        FLAGS.interpolation, 'counts_cat_{:d}.pkl.gz'.format(FLAGS.feature_i)))

    len_t = max(t for _, t in X)+1
    print("len_t", len_t)
    N_cats = len(total_counts)
    counts = np.zeros([N_cats, N_cats, len_t], dtype=np.float)

    for (cat, t), c in zip(X, y):
        counts[cat,c,t] += 1
    l = [make_training_test(counts, proportion=0.3)
         for _ in range(FLAGS.num_set_draws)]
    training_set, test_set = zip(*l)
    del l
    training_set = np.reshape(np.array(training_set, dtype=np.float32),
                              [-1, N_cats, len_t])
    test_set = np.reshape(np.array(test_set, dtype=np.float32),
                              [-1, N_cats, len_t])

    with tf.variable_scope("cat_{:d}".format(FLAGS.feature_i)):
        log_likelihood, scale = training_model(
            training_set, test_set, total_counts, len_t, N_cats)
        train_op = (tf.train.AdamOptimizer(learning_rate=0.001)
                    .minimize(-log_likelihood))

    times_waiting = 0
    prev_ll = -np.inf
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(50000):
            sess.run(train_op)
            if i%100 == 0:
                _, ll, s = sess.run([train_op, log_likelihood, scale])
                print("step", i, ll, ", scale =", s)
                if ll > prev_ll:
                    times_waiting = 0
                else:
                    times_waiting += 1
                    if times_waiting > FLAGS.patience:
                        break
                prev_ll = ll
        s = sess.run(scale)
        print("Final scale:", s)
        pu.dump(s, scale_fname)

if __name__ == '__main__':
    tf.app.run()
