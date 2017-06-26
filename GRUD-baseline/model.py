import tensorflow as tf
import numpy as np
from gru_ln_dropout_cell import (LayerNormDropoutGRUDCell,
                                 LayerNormBasicGRUCell,
                                 LayerNormVariationalDropoutGRUCell)
from slugify import slugify
import collections
import pickle_utils as pu
import os

from tensorflow.python.util import nest
from bb_alpha_inputs import add_variable_scope, model as bb_alpha_model
import fast_smooth_category_counts as fscc

def _make_embedding(_name, n_cats, index, total_counts=None):
    if total_counts is None:
        total_counts = np.array([1]*n_cats)

    name = slugify(_name, separator='_')
    ignore_categories = np.concatenate(([True], (total_counts==0)), axis=0)

    # We use the ceiling of log2(n_cats), each dimension can hold a bit.
    n_dims = int(np.ceil(np.log2(n_cats)))
    embeddings = tf.get_variable(
        name,
        shape=[n_cats+1, n_dims],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=True)
    for i, ignore in enumerate(ignore_categories):
        if ignore:
            make_nan = embeddings[i,:].assign([np.nan]*n_dims)
            tf.add_to_collection('make_embeddings_nan', make_nan)
    return tf.nn.embedding_lookup(embeddings, index+1), n_dims

def _embed_categorical(inputs, categorical_headers, number_of_categories,
                       interpolation_dir):
    "Embed categorical inputs and concatenate them with numerical inputs"
    recurrent_inputs_l = [inputs['numerical_ts']]
    recurrent_inputs_dt_l = [inputs['numerical_ts_dt']]
    static_inputs_l = [inputs['numerical_static']]

    with tf.variable_scope('embeddings'):
        for i, h in enumerate(categorical_headers):
            n_cats = number_of_categories['categorical_ts'][i]
            input_slice = inputs['categorical_ts'][:,:,i]
            dt_slice = inputs['categorical_ts_dt'][:,i:i+1]
            total_counts = pu.load(os.path.join(
                interpolation_dir, 'counts_cat_{:d}.pkl.gz'.format(i)))

            t, n_dims = _make_embedding(h, n_cats, input_slice, total_counts)
            recurrent_inputs_l.append(t)
            recurrent_inputs_dt_l.append(tf.tile(dt_slice, [1, n_dims]))


        for i, n_cats in enumerate(
                number_of_categories['categorical_static']):
            input_slice = inputs['categorical_static'][:,i]
            static_inputs_l.append(_make_embedding(
                'static_{:d}'.format(i), n_cats, input_slice)[0])
    recurrent_inputs = tf.concat(recurrent_inputs_l, axis=2)
    recurrent_inputs_dt = tf.concat(recurrent_inputs_dt_l, axis=1)
    assert recurrent_inputs.get_shape()[2] == recurrent_inputs_dt.get_shape()[1]
    static_inputs = tf.concat(static_inputs_l, axis=1)
    return recurrent_inputs, recurrent_inputs_dt, static_inputs

def _tile_static(recurrent_inputs, recurrent_inputs_dt, static_inputs):
    "Tile static input to each time step"
    static_inputs_dt = tf.zeros_like(static_inputs, dtype=tf.float32)
    tiled_static_inputs = tf.tile(tf.expand_dims(static_inputs, 1),
                                  [1, tf.shape(recurrent_inputs)[1], 1])
    inputs_dt = tf.concat([recurrent_inputs_dt, static_inputs_dt], axis=1)
    inputs = tf.concat([recurrent_inputs, tiled_static_inputs], axis=2)
    return inputs, inputs_dt


def GRUD(num_units, num_layers, inputs_dict, input_means_dict,
         number_of_categories, categorical_headers, default_batch_size,
         layer_norm, interpolation_dir):
    del interpolation_dir

    keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
    recurrent_inputs, recurrent_inputs_dt, static_inputs = _embed_categorical(
        inputs_dict, categorical_headers, number_of_categories,
        interpolation_dir)

    inputs, inputs_dt = _tile_static(recurrent_inputs, recurrent_inputs_dt,
                                     static_inputs)
    # If this were not a risk score, we could concatenate
    # `inputs_dict['time_until_label']` with `inputs` here

    # Input means will default to 0 for categorical variables
    input_means = np.zeros([inputs_dt.get_shape()[1]], dtype=np.float32)
    input_means[:inputs_dict['numerical_ts_dt'].get_shape()[1]] = (
        input_means_dict['numerical_ts'])

    # Create cell
    cells = [LayerNormDropoutGRUDCell(num_units, input_means,
            dropout_keep_prob=keep_prob, layer_norm=layer_norm)]
    for _ in range(1, num_layers):
        cells.append(LayerNormBasicGRUCell(
            num_units, dropout_keep_prob=keep_prob, layer_norm=layer_norm))
    if len(cells) > 1:
        cell = tf.contrib.rnn.MultiRNNCell(cells)
    else:
        cell = cells[0]

    # Create cell's initial state
    state_size_flat = nest.flatten(cell.state_size)
    ini_state_flat = []
    for i, sz in enumerate(state_size_flat):
        if i==1:
            # This is the part containing the time since last input
            ini_state_flat.append(inputs_dt)
        else:
            ini_state_flat.append(tf.zeros(
                [tf.shape(inputs_dict['label'])[0], sz], dtype=tf.float32))
    initial_state = nest.pack_sequence_as(structure=cell.state_size,
                                          flat_sequence=ini_state_flat)

    # Create RNN
    rnn_outputs, _ = tf.nn.dynamic_rnn(
        cell, inputs=inputs,
        sequence_length=inputs_dict['length'],
        initial_state=initial_state,
        dtype=tf.float32,
        parallel_iterations=default_batch_size*2)

    # Flatten the outputs so we can compare with the labels
    with tf.variable_scope("sequence_mask"):
        mask = tf.sequence_mask(inputs_dict['length'], tf.shape(rnn_outputs)[1])
        flat_rnn_outputs = tf.boolean_mask(rnn_outputs, mask)
        flat_labels = tf.boolean_mask(inputs_dict['label'], mask)

    with tf.variable_scope("logistic_layer"):
        weight = tf.get_variable(
            "W", dtype=tf.float32,
            shape=[cell.output_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        bias = tf.get_variable(
            "b", dtype=tf.float32,
            shape=[],
            initializer=tf.constant_initializer(0.1),
            trainable=True)
        flat_logits = tf.reduce_sum(flat_rnn_outputs * weight, axis=1) + bias
        prediction = tf.nn.sigmoid(flat_logits)

    xent_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=flat_logits, labels=flat_labels)
    loss = tf.reduce_mean(xent_loss)
    # We could add autoregression to the loss
    # As well as predicting `input_dict['treatments_ts']`

    return {'flat_risk_score': prediction,
            'loss': loss,
            'keep_prob': keep_prob,
            'flat_labels': flat_labels}

def ventilation_risk_score_raw(inputs_dict, outputs_dict):
    with tf.variable_scope("risk_score_raw"):
        ventilation_mask = tf.sequence_mask(
            inputs_dict['n_ventilations'], tf.shape(inputs_dict['ventilation_ends'])[1])
        tf.assert_non_negative(inputs_dict['ventilation_ends'])
        tf.assert_non_negative(inputs_dict['length'])
        ventilation_ends_for_flat_labels = \
            inputs_dict['ventilation_ends'] + tf.expand_dims(
                tf.cumsum(inputs_dict['length'], exclusive=True), 1)
        flat_ventilation_ends = tf.boolean_mask(
            ventilation_ends_for_flat_labels, ventilation_mask)

        labels = tf.gather(outputs_dict['flat_labels'], flat_ventilation_ends)

        return [flat_ventilation_ends+1, labels, outputs_dict['flat_risk_score']]

def ventilation_risk_score(vent_ends, labels, predictions, hours_before):
    assert hours_before == sorted(hours_before, reverse=True), \
            "hours_before must be in descending order"

    assert len(predictions.shape) == 1
    assert len(vent_ends.shape) == 1
    assert labels.shape == vent_ends.shape

    d = dict((h, ([], [])) for h in hours_before)

    prev_ve = 0
    for label, vent_end in zip(labels, vent_ends):
        for h in hours_before:
            if vent_end-h > prev_ve:
                y_true, y_score = d[h]
                y_true.append(label)
                y_score.append(predictions[prev_ve:vent_end-h].max())
        prev_ve = vent_end

    return d

import sklearn.metrics.ranking

def binary_auc_tpr_ppv(y_true, y_score, sample_weight=None):
    "Calculates TPR/PPV AUC given true y and prediction scores"
    if len(y_true) <= 1:
        return np.nan
    fps, tps, thresholds = \
        sklearn.metrics.ranking._binary_clf_curve(
            y_true, y_score, sample_weight=sample_weight)
    tpr = tps / tps[-1]
    ppv = tps / (tps + fps)
    return sklearn.metrics.ranking.auc(tpr, ppv)

def _bayes_embedding(_name, n_cats, index, total_counts, interpolation_dir,
                     category_index, cur_cat, cur_time):
    category_samples = pu.load(os.path.join(interpolation_dir,
                             'cat_{:d}.pkl.gz'.format(category_index)))
    scale = pu.load(os.path.join(interpolation_dir, 'trained',
                                 'cat_{:d}'.format(category_index),
                                 'scale.pkl'))
    p = fscc.make_smoothed_probability(category_samples, total_counts, scale)
    occurring_cats = (total_counts != 0)
    assert not np.any(np.isnan(p[occurring_cats,:,occurring_cats]))
    assert not np.any(np.isinf(p[occurring_cats,:,occurring_cats]))
    assert not np.any(p[occurring_cats,:,occurring_cats] < 0)
    p[:,:,~occurring_cats] = 0
    # Revert to baseline in categories that don't occur in the training data
    p[~occurring_cats,:,:] = total_counts/np.sum(total_counts)

    n_dims = int(np.ceil(np.log2(np.sum(occurring_cats))))

    name = slugify(_name, separator='_')
    with tf.variable_scope(name):
        need_to_sample = (cur_cat < 0)
        logits = tf.constant(np.log(p), name="logits", dtype=tf.float32)
        cur_cat_time = tf.concat([cur_cat, cur_time], axis=1)
        sampled_index = tf.to_int32(tf.multinomial(
            tf.gather_nd(logits, cur_cat_time), 1))

        index = tf.where(need_to_sample, sampled_index, cur_cat, name="index")
        assert int(index.get_shape()[1]) == 1
        assert len(index.get_shape()) == 2

        # Some of the embeddings here might never get used -- if the category
        # they belong to never appears
        embeddings = tf.get_variable(
            'embeddings',
            shape=[n_cats, n_dims],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        return tf.nn.embedding_lookup(embeddings, index), n_dims

BDCellStateTuple = collections.namedtuple("BDCellStateTuple",
                                          ("h", "c_dt", "c_1", "x_dt", "x_1"))

class BayesDropoutCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    """Unit with categorical embeddings, static data support, layer
    normalization and recurrent dropout."""
    def __init__(self, num_units, inputs, number_of_categories,
                 categorical_headers, numerical_headers, interpolation_dir,
                 recurrent_dropout, output_dropout, num_samples):
        static_inputs_l = [inputs['numerical_static']]
        with tf.variable_scope('static_embeddings'):
            for i, n_cats in enumerate(
                    number_of_categories['categorical_static']):
                input_slice = inputs['categorical_static'][:,i]
                static_inputs_l.append(_make_embedding(
                    'static_{:d}'.format(i), n_cats, input_slice)[0])
        self._static_inputs = tf.concat(static_inputs_l, axis=1)

        self._categorical_headers = categorical_headers
        self._numerical_headers = numerical_headers
        self._interpolation_dir = interpolation_dir
        self._n_categorical_ts = number_of_categories['categorical_ts']

        self._cat_n = len(self._n_categorical_ts)
        self._num_n = int(inputs['numerical_ts'].get_shape()[2])

        self._gru = LayerNormVariationalDropoutGRUCell(
            num_units, recurrent_dropout, output_dropout)
        self._num_samples = num_samples

    @property
    def state_size(self):
        return BDCellStateTuple(self._gru.state_size, self._cat_n, self._cat_n,
                                self._num_n, self._num_n)
    @property
    def output_size(self):
        return self._gru.output_size

    def _bayes_num(self, name, interpolation_dir, num_i, prev_x, prev_dt):
        inputs = tf.concat([prev_x, prev_dt], axis=1,
                           name="concat_{:s}".format(name))
        mX, sX, my, sy, N = pu.load(os.path.join(
            interpolation_dir, 'trained', 'num_{:d}'.format(num_i),
            'means.pkl.gz'))
        m = bb_alpha_model(inputs, labels=None, N=N,
              num_samples=self._num_samples, layer_sizes=[64], alpha=0.5,
              trainable=False, mean_X=mX, mean_y=my, std_X=sX, std_y=sy,
              name=name)
        return m['samples']

    def _accumulate_values(prev_dt, prev_1, inputs, inputs_cond):
        input_zeros = tf.zeros_like(inputs, dtype=tf.float32)
        dt = tf.where(inputs_cond, prev_dt+1, input_zeros)
        ins = tf.where(inputs_cond, prev_1, inputs)
        return dt, ins

    def __call__(self, inputs, state, scope=None):
        prev_h, prev_c_dt, prev_c, prev_x_dt, prev_x = state
        categorical_ts, numerical_ts = inputs

        with tf.variable_scope(scope or "bayes_dropout_cell"):
            to_concat = []
            with tf.variable_scope("embeddings"):
                for i, name in enumerate(self._categorical_headers):
                    total_counts = pu.load(os.path.join(
                        self._interpolation_dir,
                        'counts_cat_{:d}.pkl.gz'.format(i)))
                    to_concat.append(_bayes_embedding(
                        name, self._n_categorical_ts[i], categorical_ts[:,i],
                        total_counts, self._interpolation_dir, i,
                        prev_c[:,i:i+1], prev_c_dt[:,i:i+1]))
            with tf.variable_scope("num_inputs"):
                for i, name in enumerate(self._numerical_headers):
                    l = self._bayes_num(
                        name, self._interpolation_dir, i,
                        prev_x[:,i:i+1], prev_x_dt[:,i:i+1])
                    num_slice = numerical_ts[:,i]
                    num_is_nan = tf.isnan(num_slice)
                    l = tf.where(num_is_nan, l, num_slice)
                    to_concat.append(l)
            to_concat.append(self._static_inputs)
            all_inputs = tf.concat(to_concat, axis=1, name="all_inputs")

            with tf.variable_scope("accumulate_num"):
                x_dt, x_1 = self._accumulate_values(
                    prev_x_dt, prev_x_1, numerical_ts, tf.is_nan(numerical_ts))
            with tf.variable_scope("accumulate_cat"):
                c_dt, c_1 = self._accumulate_values(
                    prev_c_dt, prev_c_1, categorical_ts, (categorical_ts < 0))
            h = self._gru(inputs=all_inputs, state=prev_h)
        return h, BDCellStateTuple(h, c_dt, c_1, x_dt, x_1)


def BayesDropout(num_units, num_layers, inputs_dict, input_means_dict,
                 number_of_categories, categorical_headers, numerical_headers,
                 default_batch_size, layer_norm, interpolation_dir, num_samples):
    del input_means_dict

    if num_layers != 1:
        raise NotImplementedError("more than 1 layer")

    keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
    output_dropout = tf.floor(keep_prob + tf.random_uniform(
        [default_batch_size, num_units]), name="output_dropout")
    recurrent_dropout = tf.floor(keep_prob + tf.random_uniform(
        [default_batch_size, num_units]), name="recurrent_dropout")

    cell = BayesDropoutCell(num_units=num_units,
                            inputs=inputs_dict,
                            number_of_categories=number_of_categories,
                            categorical_headers=categorical_headers,
                            numerical_headers=numerical_headers,
                            interpolation_dir=interpolation_dir,
                            recurrent_dropout=recurrent_dropout,
                            output_dropout=output_dropout,
                            num_samples=num_samples)

    # Create cell's initial state
    state_size_flat = nest.flatten(cell.state_size)
    ini_state_flat = []
    for i, sz in enumerate(state_size_flat):
        if i == 1: # Time since last input (categories)
            ini_state_flat.append(tf.cast(
                inputs_dict['categorical_ts_dt'], tf.int32))
        elif i == 3: # Time since last input (categories)
            ini_state_flat.append(inputs_dict['numerical_ts_dt'])
        else:
            if i == 2:
                t = tf.int32
            else:
                t = tf.float32
            ini_state_flat.append(tf.zeros(
                [tf.shape(inputs_dict['label'])[0], sz], dtype=t))
    initial_state = nest.pack_sequence_as(structure=cell.state_size,
                                          flat_sequence=ini_state_flat)

    rnn_outputs, _ = tf.nn.dynamic_rnn(
        cell, inputs=(inputs_dict['categorical_ts'], inputs_dict['numerical_ts']),
        sequence_length=inputs_dict['length'],
        initial_state=initial_state,
        dtype=tf.float32,
        parallel_iterations=default_batch_size*2)


    #### COPIED FROM GRUD

    # Flatten the outputs so we can compare with the labels
    with tf.variable_scope("sequence_mask"):
        mask = tf.sequence_mask(inputs_dict['length'], tf.shape(rnn_outputs)[1])
        flat_rnn_outputs = tf.boolean_mask(rnn_outputs, mask)
        flat_labels = tf.boolean_mask(inputs_dict['label'], mask)

    with tf.variable_scope("logistic_layer"):
        weight = tf.get_variable(
            "W", dtype=tf.float32,
            shape=[cell.output_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        bias = tf.get_variable(
            "b", dtype=tf.float32,
            shape=[],
            initializer=tf.constant_initializer(0.1),
            trainable=True)
        flat_logits = tf.reduce_sum(flat_rnn_outputs * weight, axis=1) + bias
        prediction = tf.nn.sigmoid(flat_logits)

    xent_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=flat_logits, labels=flat_labels)
    loss = tf.reduce_mean(xent_loss)
    # We could add autoregression to the loss
    # As well as predicting `input_dict['treatments_ts']`

    return {'flat_risk_score': prediction,
            'loss': loss,
            'keep_prob': keep_prob,
            'flat_labels': flat_labels}
