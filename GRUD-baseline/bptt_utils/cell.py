"""
Contains classes and functions useful for running BPTT when the sequences have
very different sizes.
"""

import tensorflow as tf

class ResetStateCellWrapper(tf.contrib.rnn.RNNCell):
    """
    Wraps a RNN cell in another one, that resets the wrapped cell's state when
    a time step is marked as such. It takes a tuple of two elements, the first
    is a Tensor of size (batch_size x 1), which is 0 if the step doesn't need
    resetting, and 1 if it does.

    To mark a time-step as needing resetting, make
    its first element be non-zero.
    """
    def __init__(self, cell, batch_size, dtype=tf.float32):
        self._cell = cell
        self._zero_state = cell.zero_state(batch_size, dtype=dtype)

    @property
    def state_size(self):
        return self._cell.state_size
    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        def replace(condition, t, f):
            "tf.where for a nested named tuple of tensors"
            if isinstance(t, tf.Tensor):
                assert isinstance(f, tf.Tensor)
                return tf.where(condition, t, f)
            elems = (replace(condition, i, j) for i, j in zip(t, f))
            return t.__class__(*elems)

        needs_reset = inputs[:,0]
        with tf.variable_scope(scope or 'wrapped'):
            _state = replace(tf.equal(needs_reset, 0.0),
                             state, self._zero_state)
            return self._cell(inputs[:,1:], _state)
