import tensorflow as tf
import numpy as np
import math
from gru_ln_dropout_cell import LayerNormDropoutGRUDCell, LayerNormBasicGRUCell

def GRUD(self, num_units, num_layers, input_means, inputs, training_keep_prob,
         learning_rate, batch_size, layer_norm):

    cells = [LayerNormDropoutGRUDCell(num_units, input_means,
            dropout_keep_prob=self.keep_prob, layer_norm=layer_norm)]
    for _ in range(1, num_layers):
        cells.append(LayerNormBasicGRUCell(
            num_units, dropout_keep_prob=self.keep_prob, layer_norm=layer_norm))
    if len(cells) > 1:
        cell = tf.contrib.rnn.MultiRNNCell(cells)
    else:
        cell = cells[0]

    rnn_outputs, next_state = tf.nn.dynamic_rnn(
        cell,
        inputs=inputs['numerical_ts'],
        sequence_length=inputs['length'],
        initial_state=cell.zero_state(dtype=tf.float32)
        dtype=tf.float32)

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
        logit = tf.reduce_sum(last_outputs * self.W, axis=1) + self.b
    xent_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.labels)
    self.loss = tf.reduce_mean(xent_loss)
    self.pred = tf.nn.sigmoid(logits)

    return tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)
