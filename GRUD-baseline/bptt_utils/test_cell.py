import unittest
import numpy as np
import tensorflow as tf

from cell import ResetStateCellWrapper

class TestCell(tf.contrib.rnn.RNNCell):
    state_size = tf.contrib.rnn.LSTMStateTuple(1, 1)
    output_size = 4

    def __init__(self):
        pass

    def __call__(self, inputs, state, scope=None):
        n_state = tf.contrib.rnn.LSTMStateTuple(*(i+1 for i in state))
        return tf.concat(list(n_state)+[inputs], axis=1), n_state

class TestResetStateCellWrapper(unittest.TestCase):
    def test_reset_state(self):
        inputs = tf.contrib.rnn.LSTMStateTuple(
            tf.constant([[[0], [0], [0], [0], [1]]], dtype=tf.float32),
            tf.constant([[[2,-2], [3,-3], [3,-3], [-2,2], [0,3]]], dtype=tf.float32))
        cell = ResetStateCellWrapper(TestCell(), batch_size=1)
        outputs, _next_state = tf.nn.dynamic_rnn(
            cell, inputs=inputs,
            initial_state=cell.zero_state(batch_size=1, dtype=tf.float32),
            dtype=tf.float32)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run([outputs], {})[0]
        true_result = np.array([[
            [ 1., 1., 2., -2.],
            [ 2., 2., 3., -3.],
            [ 3., 3., 3., -3.],
            [ 4., 4., -2., 2.],
            [ 1., 1., 0., 3.]]], dtype=np.float32)
        self.assertTrue(np.all(result == true_result))


if __name__ == '__main__':
    unittest.main()
