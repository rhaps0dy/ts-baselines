import numpy as np
import tensorflow as tf

def build_dt(d, d_valid, initial_dt, also_delayed=False):
    d = np.transpose(d, [1,0,2])
    d_valid = np.transpose(d_valid, [1,0,2])
    dt = np.zeros_like(d, dtype=np.int64)
    dt[0,:,:] = initial_dt
    if also_delayed:
        delayed_dt = np.copy(dt)
        delayed_dt[0,:,:] = initial_dt
        delayed_dt[0,(initial_dt==0)] = 1000
    for t in range(1, dt.shape[0]):
        dt[t] = dt[t-1] + 1
        if also_delayed:
            delayed_dt[t] = dt[t]
        dt[t,d_valid[t]] = 0

    dt = np.transpose(dt, [1,0,2])
    if also_delayed:
        return dt, np.transpose(delayed_dt, [1,0,2])
    return dt

def build_impute_forward(d, d_valid, means, also_delayed=False):
    batch_size = d.shape[0]
    d = np.transpose(d, [1,0,2])
    d_valid = np.transpose(d_valid, [1,0,2])
    impute = np.copy(d)
    initial_valid = d_valid[0,:]
    batch_means = np.stack([means]*batch_size)
    impute[0,initial_valid] = batch_means[initial_valid]
    for t in range(1, impute.shape[0]):
        to_carry_forward = ~d_valid[t,:]
        impute[t,to_carry_forward] = impute[t-1,to_carry_forward]
    impute_transposed = np.transpose(impute, [1,0,2])
    if also_delayed:
        delayed_impute = np.empty(impute.shape, dtype=impute.dtype)
        delayed_impute[1:,:,:] = impute[:-1,:,:]
        delayed_impute[0,:,:] = batch_means
        return impute_transposed, np.transpose(delayed_impute, [1,0,2])
    return impute_transposed

def make_within_length(d, length):
    batch_sz, len_t, n_cats = d.shape
    time = np.stack([np.arange(len_t)] * batch_sz)[:,:,None]
    length = length[:,None,None]
    assert list(time.shape) == [batch_sz, len_t, 1]
    assert list(length.shape) == [batch_sz, 1, 1]
    return (time < length)


EXAMPLE = {
    'context': {
        'icustay_id': 'int64',
        'numerical_static': 'float',
        'categorical_static': 'int64',
        'numerical_ts_dt': 'float',
        'categorical_ts_dt': 'float',
    },
    'sequence': {
        'time_until_label': 'float',
        'label': 'float',

        'numerical_ts': 'float',
#        'numerical_ts_forward': 'float',
#        'numerical_ts_dt_all': 'int64',

        'categorical_ts': 'int64',
#        'categorical_ts_forward': 'int64',
#        'categorical_ts_dt_all': 'int64',

        'treatments_ts': 'float',
        'ventilation_ends': 'int64',
    },
}

def _feature(feature, key, dtype, iterable):
    a = getattr(feature.feature[key], dtype+'_list').value
    a.extend(iterable)

def context(example, key, dtype, iterable):
    return _feature(example.context, key, dtype, iterable)
def feature(example, key, dtype, iterable):
    return _feature(example.features, key, dtype, iterable)

def sequence(example, key, dtype, iterable):
    feature_list = example.feature_lists.feature_list[key].feature
    for row in iterable.reshape([len(iterable), -1]):
        a = getattr(feature_list.add(), dtype+'_list').value
        a.extend(row)

def example_from_data(data):
    funs = {'context': context, 'sequence': sequence}
    example = tf.train.SequenceExample()
    for kind in EXAMPLE:
        for name, tp in EXAMPLE[kind].items():
            if name not in data:
                print("WARNING: no key `{:s}` present in `data`"
                      .format(name))
            else:
                funs[kind](example, name, tp, data[name])
    return example
