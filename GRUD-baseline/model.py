import tensorflow as tf
import numpy as np
import math
from gru_ln_dropout_cell import LayerNormDropoutGRUDCell, LayerNormBasicGRUCell
from slugify import slugify
import collections

from tensorflow.python.util import nest

def _make_embedding(_name, n_cats, index):
    name = slugify(_name, separator='_')

    # We use the ceiling of log2(n_cats), each dimension can hold a bit.
    n_dims = int(math.ceil(np.log2(n_cats)))
    embeddings = tf.get_variable(
        name,
        shape=[n_cats+1, n_dims],
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(), # TODO: make NaN
        trainable=True)
    make_nan = embeddings[0,:].assign([np.nan]*n_dims)
    tf.add_to_collection('make_embeddings_nan', make_nan)
    return tf.nn.embedding_lookup(embeddings, index+1), n_dims

def _embed_categorical(inputs, categorical_headers, number_of_categories):
    "Embed categorical inputs and concatenate them with numerical inputs"
    recurrent_inputs_l = [inputs['numerical_ts']]
    recurrent_inputs_dt_l = [inputs['numerical_ts_dt']]
    static_inputs_l = [inputs['numerical_static']]

    _used_names = set()
    with tf.variable_scope('embeddings'):
        for i, h in enumerate(categorical_headers):
            n_cats = number_of_categories['categorical_ts'][i]
            input_slice = inputs['categorical_ts'][:,:,i]
            dt_slice = inputs['categorical_ts_dt'][:,i:i+1]

            # Workaround for duplicated names
            while h in _used_names:
                h += '1'
            _used_names.add(h)
            ###

            t, n_dims = _make_embedding(h, n_cats, input_slice)
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
         layer_norm):

    keep_prob = tf.placeholder(shape=[], dtype=tf.float32)
    recurrent_inputs, recurrent_inputs_dt, static_inputs = _embed_categorical(
        inputs_dict, categorical_headers, number_of_categories)

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

# Calculate AUC with prediction
import sklearn.metrics.ranking

def ventilation_risk_score(inputs_dict, outputs_dict, hours_before):
    def _ventilation_risk_score(vent_ends, labels, predictions):
        assert len(predictions.shape) == 1
        assert len(vent_ends.shape) == 1
        assert labels.shape == vent_ends.shape

        _hours_before = hours_before[:] # copy
        _hours_before.sort(reverse=True)

        d = dict((h, ([], [])) for h in _hours_before)

        prev_ve = 0
        for label, vent_end in zip(labels, vent_ends):
            for h in _hours_before:
                if vent_end-h > prev_ve:
                    l1, l2 = d[h]
                    l1.append(predictions[prev_ve:vent_end-h].max())
                    l2.append(label)
            prev_ve = vent_end

        def _binary_auc_tpr_ppv(y_true, y_score, sample_weight=None):
            if len(y_true) <= 1:
                return np.nan
            tps, fps, thresholds = \
                sklearn.metrics.ranking._binary_clf_curve(
                    y_true, y_score, sample_weight=sample_weight)
            tpr = tps / sum(y_true)
            ppv = tps / (tps + fps)
            print("shapes:", tpr.shape, ppv.shape)
            return sklearn.metrics.ranking.auc(tpr, ppv, reorder=True)

        for hour, (predictions, labels) in d.items():
            d[hour] = _binary_auc_tpr_ppv(labels, predictions)
        return list(d[h] for h in hours_before)


    with tf.variable_scope("tpr_ppv"):
        ventilation_mask = tf.sequence_mask(
            inputs_dict['n_ventilations'], tf.shape(inputs_dict['ventilation_ends'])[1])
        ventilation_ends_for_flat_labels = \
            inputs_dict['ventilation_ends'] + tf.expand_dims(
                tf.cumsum(inputs_dict['length'], exclusive=True), 1)
        flat_ventilation_ends = tf.boolean_mask(
            ventilation_ends_for_flat_labels, ventilation_mask)

        labels = tf.gather(outputs_dict['flat_labels'], flat_ventilation_ends)

        return [flat_ventilation_ends+1, labels, outputs_dict['flat_risk_score']], _ventilation_risk_score #tf.py_func(
            #_ventilation_risk_score,
            #,
            #Tout=[tf.float32]*len(hours_before),
            #stateful=False)
