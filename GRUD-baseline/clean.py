"""
Creates a list of Examples and saves them to "clean.pkl.gz"
"""
import collections
NpExample = collections.namedtuple('NpExample', [
    'icustay_id', 'ventilation_ends', 'time_until_label', 'label',
    'numerical_static', 'categorical_static', 'numerical_ts', 'categorical_ts',
    'treatments_ts'])


import numpy as np
import pandas as pd
import itertools as it
import pickle_utils as pu
import gzip
import csv
import math
from tqdm import tqdm

def get_headers(table):
    with gzip.open('../../{:s}.csv.gz'.format(table), 'rt',
            newline='') as csvf:
        return next(iter(csv.reader(csvf)))[3:]

def determine_type(header, b_is_category):
    t = header[0]
    if t == 'C':
        return "category"
    elif t == 'B':
        return "category" if b_is_category else np.bool
    elif t == 'F':
        return np.float32
    else:
        raise ValueError(header)

@pu.memoize('parsed_dataframe_{n_frequent:d}.pkl.gz', log_level='warn')
def parse_csv_frequent_headers(n_frequent):
    zero_headers = get_headers('outputevents')
    bool_headers = (get_headers('procedureevents_mv') +
                    get_headers('drugevents'))
    nan_headers = get_headers('labevents') + get_headers('chartevents')

    dtype = dict(it.chain(
        map(lambda e: (e, determine_type(e, True)), nan_headers + zero_headers),
        map(lambda e: (e, determine_type(e, False)), bool_headers),
        map(lambda e: (e, np.int32), ["icustay_id", "hour", "subject_id"])))
    fillna = dict(it.chain(
        map(lambda e: (e, 0.0), zero_headers),
        map(lambda e: (e, False), bool_headers)))

    count_headers, _ = pu.load("../../mimic-clean/number_non_missing.pkl.gz")
    count_headers.sort()
    _, headers = zip(*count_headers)
    non_data_headers = set(bool_headers + ['icustay_id', 'hour', 'subject_id'])
    # Remove headers corresponding to treatments and indices
    data_headers = filter(lambda h: h not in non_data_headers, headers)
    data_headers = list(data_headers)[-n_frequent:]
    # Separate headers that are categorical
    categorical_headers = list(filter(lambda h: dtype[h] == 'category', data_headers))
    numerical_headers = list(filter(lambda h: dtype[h] != 'category', data_headers))

    usecols = list(set(['icustay_id', 'hour', 'B pred last_ventilator'] +
                       bool_headers + categorical_headers + numerical_headers))
    del fillna['B pred last_ventilator']

    df = pd.read_csv('../../mimic.csv.gz', header=0, index_col='icustay_id',
                usecols=usecols, dtype=dtype, engine='c', true_values=[b'1'],
                false_values=[b'0', b''])
    return df, fillna, numerical_headers, categorical_headers, bool_headers

@pu.memoize('static_data.pkl.gz', log_level='warn')
def get_static_data():
    numerical = ["r_admit_time", "b_gender", "r_age", "i_previous_admissions",
               "i_previous_icustays"]
    categorical = ["c_admit_type", "c_admit_location", "c_ethnicity"]
    usecols = ["icustay_id"] + numerical + categorical

    dtype = dict(it.chain(
        zip(numerical, it.repeat(np.float32)),
        zip(categorical, it.repeat('category'))))
    df = pd.read_csv('../../static_patients.csv.gz', header=0, index_col="icustay_id",
                     usecols=usecols, dtype=dtype, engine='c',
                     true_values=[b'1'], false_values=[b'0', b''])
    df.r_admit_time = df.r_admit_time.apply(lambda n: n/3600)
    return df[numerical], df[categorical].applymap(int)

@pu.memoize('clean_{n_frequent:d}.pkl.gz', log_level='warn')
def clean(n_frequent):
    mimic, fillna, numerical_headers, categorical_headers, treatment_headers = \
        parse_csv_frequent_headers(n_frequent=n_frequent)
    static_data_numerical, static_data_categorical = get_static_data()

    # Discard patients who have any CV data, and fill
    #usable_hadm_ids, icustay_hadmid_drift = pu.load('../../mimic-clean/db_things.pkl.gz')
    #mimic = mimic.select((lambda icustay_id: icustay_hadmid_drift[icustay_id][1] in
    #                      usable_hadm_ids), axis=0).fillna(fillna)
    mimic = mimic.fillna(fillna)

    examples = []
    for icustay_id, df in tqdm(mimic.groupby(level=0)):
        ventilation_ends = df[['hour', 'B pred last_ventilator']].dropna().values
        if len(ventilation_ends) == 0:
            continue
        ts_len = ventilation_ends[-1,0]+1

        time_until_label = np.empty([ts_len, 1], dtype=np.uint16)
        for hour, _ in reversed(ventilation_ends):
            time_until_label[:hour+1, 0] = np.arange(hour, -1, -1, dtype=np.uint16)

        # Label is 1 when the patient will be re-ventilated
        label = np.zeros([ts_len, 1], dtype=np.bool)
        if len(ventilation_ends) > 1:
            label[:ventilation_ends[-2,0]+1, 0] = 1

        hours = df['hour']

        numerical_ts = np.empty([ts_len, len(numerical_headers)], dtype=np.float32)
        numerical_ts[...] = np.nan
        numerical_ts_dt = np.zeros([len(numerical_headers)], dtype=np.uint16) + 1000
        numerical_arr = df[numerical_headers].values

        categorical_ts = -np.ones([ts_len, len(categorical_headers)], dtype=np.int8)
        categorical_ts_dt = np.zeros([len(numerical_headers)], dtype=np.uint16) + 1000
        categorical_df = df[categorical_headers]
        categorical_df_usable = ~(_cat_df.isnull().values)


        treatments_ts = np.zeros([ts_len, len(treatment_headers)], dtype=np.bool)
        treatments_df = df[treatment_headers].values

        i=0
        while i < len(hours) and hours[i] < ts_len:
            overwrite = ~np.isnan(numerical_arr[i])
            if hours[i] <= 0:
                numerical_ts_dt[overwrite] = -hours[i]
                categorical_ts_dt[categorical_df_usable[i]] = -hours[i]

            if hours[i] >= 0:
                treatments_ts[hours[i]] = treatments_df[i]
                categorical_ts[hours[i], categorical_df_usable[i]] = \
                    categorical_df[i, categorical_df_usable[i]].applymap(int).values
                numerical_ts[0, overwrite] = numerical_arr[i, overwrite]

            i += 1

        numerical_static = static_data_numerical.loc[icustay_id].values
        assert len(numerical_static) == 1
        categorical_static = static_data_categorical.loc[icustay_id].values
        assert len(categorical_static) == 1

        examples.append(Example(icustay_id, ventilation_ends, time_until_label,
                                label, numerical_static[0], categorical_static[0],
                                numerical_ts, numerical_ts_dt, categorical_ts, categorical_ts_dt, treatments_ts))
        if len(examples) > 10:
            break
    return examples

def to_tensorflow(examples):
    import tensorflow as tf

    np.random.shuffle(examples)
    with tf.python_io.TFRecordWriter("hour_ventilation.tfrecords") as writer:
        for e in tqdm(examples):
            icustay_id, ventilation_ends, time_until_label, label, numerical_static, categorical_static, numerical_ts, categorical_ts, treatments_ts
            example = tf.train.Example(
                features = tf.train.Features(
                    feature={
                        'icustay_id': tf.train.Int64List(value=[,
                        }))
            writer.write(example.SerializeToString())



if __name__ == '__main__':
    c = clean(n_frequent=200)
