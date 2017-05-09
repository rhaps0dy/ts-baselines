import tensorflow as tf
import numpy as np
import math
from gru_ln_dropout_cell import LayerNormDropoutGRUDCell, LayerNormBasicGRUCell

import logging

log = logging.getLogger(__name__)

class GRUD:
    def __init__(self, num_units, num_layers, input_means, training_keep_prob,
                 bptt_length, batch_size, n_classes, layer_norm):
        assert n_classes==2, "multiclass not implemented"
        self.training_keep_prob = training_keep_prob
        self.max_batch_size = -1
        self.default_batch_size = batch_size
        self.input_means = input_means

        self.inputs = tf.placeholder(tf.float32, shape=[None, bptt_length, len(input_means)], name="inputs")
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        self.sequence_length = tf.placeholder(tf.int32, shape=[None], name="sequence_length")
        self.learning_rate_ph = tf.placeholder(tf.float32, shape=[])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])

        cells = [LayerNormDropoutGRUDCell(num_units, input_means,
                dropout_keep_prob=self.keep_prob, layer_norm=layer_norm)]
        for _ in range(1, num_layers):
            cells.append(LayerNormBasicGRUCell(num_units, dropout_keep_prob=self.keep_prob, layer_norm=layer_norm))
        if len(cells) > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell(cells)
        else:
            self.cell = cells[0]
        initial_state_tf = self.create_initial_state_placeholder()
        if isinstance(initial_state_tf[0], tuple):
            self.dt_state = initial_state_tf[0].dt
            self.prev_input_state = initial_state_tf[0].x_1
        else:
            self.dt_state = initial_state_tf.dt
            self.prev_input_state = initial_state_tf.x_1

        self.rnn_outputs, next_state = tf.nn.dynamic_rnn(
            self.cell,
            inputs=self.inputs,
            sequence_length=self.sequence_length,
            initial_state=initial_state_tf,
            dtype=tf.float32)
        self.create_next_state_tensor_list(next_state)

        inds = tf.stack([tf.range(tf.shape(self.sequence_length)[0]),
                         self.sequence_length-1], axis=1)
        last_outputs = tf.gather_nd(self.rnn_outputs, inds)

        with tf.variable_scope("softmax_layer"):
            self.W = tf.get_variable("W", dtype=tf.float32,
                                     shape=[self.cell.output_size],
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)
            self.b = tf.get_variable("b", dtype=tf.float32,
                                     shape=[],
                                     initializer=tf.constant_initializer(0.1),
                                     trainable=True)
            logits = tf.reduce_sum(last_outputs * self.W, axis=1) + self.b
        xent_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.labels)
        self.loss = tf.reduce_mean(xent_loss)
        self.pred = tf.nn.sigmoid(logits)

    def train_step(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)

    def create_initial_state_placeholder(self):
        self.initial_state = []
        initial_state_tf = []
        state_tuple = isinstance(self.cell, tf.contrib.rnn.MultiRNNCell)
        if state_tuple:
            sizes = self.cell.state_size
        else:
            sizes = [self.cell.state_size]

        for j, sz in enumerate(sizes):
            if isinstance(sz, int):
                sizes = (sz,)
            else:
                sizes = tuple(sz)
            l = []
            for i, s in enumerate(sizes):
                ph = tf.placeholder(tf.float32, shape=[None, s],
                    name='initial_state_{:d}_{:d}'.format(j, i))
                self.initial_state.append(ph)
                l.append(ph)
            if len(l) > 1:
                initial_state_tf.append(sz.__class__(*l))
            else:
                initial_state_tf.append(l[0])

        if state_tuple:
            return tuple(initial_state_tf)
        else:
            return initial_state_tf[0]

    def feed_dict(self, training_example, training=True, learning_rate=0.001):
        X, initial_dt, seq_len, labels = training_example
        l = len(labels)
        assert self.state[0].shape[0] >= l, (
               "Batch size increased from {:d} to {:d}".format(
                   self.state[0].shape[0], l))
        d = {self.inputs: X,
             self.labels: labels,
             self.dt_state: initial_dt,
             self.prev_input_state: [self.input_means]*l,
             self.sequence_length: seq_len,
             self.learning_rate_ph: learning_rate,
             self.keep_prob: self.training_keep_prob if training else 1.0}
        for i, s in enumerate(self.initial_state):
            if s is not self.dt_state and s is not self.prev_input_state:
                d[s] = self.state[i][:l]
        return d

    def create_next_state_tensor_list(self, next_state):
        self.next_state = []
        for ns in next_state:
            if isinstance(ns, tuple):
                self.next_state += ns
            else:
                self.next_state.append(ns)

    def new_epoch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.default_batch_size
        if batch_size > self.max_batch_size:
            log.warning("Increased batch size to {:d}".format(batch_size))
            self.initial_state_value = []
            max_len = max(s.get_shape()[1] for s in self.initial_state)
            z = np.zeros([batch_size, max_len], dtype=np.float32)
            for s in self.initial_state:
                self.initial_state_value.append(z[:,:s.get_shape()[1]])
            self.max_batch_size = batch_size
        self.state = list(s[:batch_size] for s in self.initial_state_value)
