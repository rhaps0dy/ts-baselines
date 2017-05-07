import tensorflow as tf

from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
_checked_scope = core_rnn_cell_impl._checked_scope # pylint: disable=protected-access

class LayerNormDropoutGRUDCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    """GRU-D unit with layer normalization and recurrent dropout."""

    def __init__(self, num_units, input_means, **kwargs):
        assert len(input_means.shape) == 1
        self._input_means = input_means
        self._input_size = len(input_means)
        kwargs['input_size'] = None
        super(LayerNormDropoutGRUDCell, self).__init__(num_units, **kwargs)

    @property
    def state_size(self):
        # Need to store the memory and the time since last value
        return core_rnn_cell.LSTMStateTuple(self._num_units, self._input_size)

    def _linear(self, inputs, each_out_size, n_outs, name, is_diagonal=False):
        if len(inputs) > 1:
            args = tf.concat(inputs, 1)
        else:
            args = inputs
        with tf.variable_scope(name):
            if is_diagonal:
                assert n_outs == 1 and each_out_size == args.get_shape()[-1]
                weights = tf.get_variable("weights", [each_out_size], dtype=tf.float32)
                out = args * weights
            else:
                out_size = n_outs*each_out_size
                proj_size = args.get_shape()[-1]
                weights = tf.get_variable("weights", [proj_size, out_size], dtype=tf.float32)
                out = tf.matmul(args, weights)

            if not self._layer_norm:
                bias = tf.get_variable("biases", [out_size], dtype=tf.float32)
                out = tf.nn.bias_add(out, bias)

            if n_outs > 1:
                out = tf.split(value=out, num_or_size_splits=n_outs, axis=1)
        return out

    def __call__(self, inputs, state, scope=None):
        """LSTM cell with layer normalization and recurrent dropout."""
        assert self._input_size == int(inputs.get_shape()[-1])

        with _checked_scope(self, scope or "layer_norm_dropout_grud_cell",
                                                reuse=self._reuse):
            prev_h, prev_dt = state
            input_zeros = tf.zeros_like(inputs, dtype=tf.float32)
            input_ones = tf.ones_like(inputs, dtype=tf.float32)
            input_nan = tf.is_nan(inputs)
            m = tf.select(input_nan, input_ones, input_zeros)
            dt = tf.select(input_nan, prev_dt+1., input_zeros)

            input_decay = self._linear([dt], self._input_size, 1,
                    "input_decay", is_diagonal=True)
            input_decay = tf.exp(-tf.nn.relu(input_decay))
            dec_inputs = tf.select(input_nan, inputs,
                    input_decay*inputs + (1-input_decay)*tf.constant(self._input_means))
            hidden_decay = self._linear([dt], self._num_units, 1, "hidden_decay")
            input_decay = tf.exp(-tf.nn.relu(hidden_decay))
            prev_h = prev_h*input_decay

            z, r = self._linear([dec_inputs, m, prev_h],
                                self._num_units, 2, "gates_1")
            if self._layer_norm:
                z = self._norm(z, "update")
                r = self._norm(r, "reset")
            z = tf.sigmoid(z)
            r = tf.sigmoid(r)

            g = self._linear([inputs, m, r*prev_h],
                             self._num_units, 1, "gates_2")
            if self._layer_norm:
                g = self._norm(g, "pre_state")
            g = self._activation(g)

            if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
                g = tf.nn.dropout(g, self._keep_prob, seed=self._seed)
            h = (1-z)*prev_h + z*g

            state = core_rnn_cell.LSTMStateTuple(h, dt)
        return h, state
