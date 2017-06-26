"""
Creates a list of Examples and saves them to "dataset/*.tfrecords"

NpExample = collections.namedtuple('NpExample', [
    'icustay_id', 'ventilation_ends', 'time_until_label', 'label',
    'numerical_static', 'categorical_static', 'numerical_ts',
    'numerical_ts_dt', 'categorical_ts', 'categorical_ts_dt', 'treatments_ts'])
"""
import collections

import numpy as np
import pandas as pd
import itertools as it
import pickle_utils as pu
import tensorflow as tf
import gzip
import csv
import math
import os
import os.path
from tqdm import tqdm

N_PROCESSORS = 8
MIMIC_PATH = '../../mimic.csv.gz'
STATIC_PATH = '../../static_patients.csv.gz'

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
        'numerical_ts_forward': 'float',
        'numerical_ts_dt_all': 'int64',
        'numerical_ts_forward_delayed': 'float',
        'numerical_ts_dt_all_delayed': 'int64',

        'categorical_ts': 'int64',
        'categorical_ts_forward': 'int64',
        'categorical_ts_dt_all': 'int64',
        'categorical_ts_forward_delayed': 'int64',
        'categorical_ts_dt_all_delayed': 'int64',

        'treatments_ts': 'float',
        'ventilation_ends': 'int64',
    },
}

def _context(example, key, dtype, iterable):
    a = getattr(example.context.feature[key], dtype+'_list').value
    a.extend(iterable)
def _sequence(example, key, dtype, iterable):
    feature_list = example.feature_lists.feature_list[key].feature
    for row in iterable.reshape([len(iterable), -1]):
        a = getattr(feature_list.add(), dtype+'_list').value
        a.extend(row)

def example_from_data(data):
    funs = {'context': _context, 'sequence': _sequence}
    example = tf.train.SequenceExample()
    for kind in EXAMPLE:
        for name, tp in EXAMPLE[kind].items():
            if name not in data:
                print("WARNING: no key `{:s}` present in `data`"
                      .format(name))
            else:
                funs[kind](example, name, tp, data[name])
    return example


def get_headers(table):
    "Get the headers of a MIMIC sub-table"
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

@pu.memoize('dataset/number_non_missing.pkl.gz', log_level='warn')
def number_non_missing():
    with gzip.open(MIMIC_PATH, 'rt') as gzf:
        f = csv.reader(gzf)
        headers = next(f)
        non_missing = np.zeros(len(headers), dtype=np.int32)
        n_lines = 0
        for line in tqdm(f):
            n_lines += 1
            for i, e in enumerate(line):
                if e!='':
                    non_missing[i] += 1
    n_missing_headers = list(zip(non_missing, headers))
    return n_missing_headers, n_lines


@pu.memoize('dataset/headers_{n_frequent:d}.pkl.gz', log_level='warn')
def get_frequent_headers(n_frequent):
    "Parses the MIMIC csv, grabbing only the `n_frequent` most frequent headers"
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

    count_headers, _ = number_non_missing()
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
    return usecols, dtype, fillna, numerical_headers, categorical_headers, bool_headers

@pu.memoize('dataset/parsed_csv.pkl.gz', log_level='warn')
def parse_csv(usecols, dtype):
    return pd.read_csv(MIMIC_PATH, header=0, index_col='icustay_id',
                       usecols=usecols, dtype=dtype, engine='c',
                       true_values=[b'1'], false_values=[b'0', b''])

@pu.memoize('dataset/static_data.pkl.gz', log_level='warn')
def get_static_data():
    numerical = ["r_admit_time", "b_gender", "r_age", "i_previous_admissions",
               "i_previous_icustays"]
    categorical = ["c_admit_type", "c_admit_location", "c_ethnicity"]
    usecols = ["icustay_id"] + numerical + categorical

    dtype = dict(it.chain(
        zip(numerical, it.repeat(np.float32)),
        zip(categorical, it.repeat('category'))))
    df = pd.read_csv(STATIC_PATH, header=0, index_col="icustay_id",
                     usecols=usecols, dtype=dtype, engine='c',
                     true_values=[b'1'], false_values=[b'0', b''])
    df.r_admit_time = df.r_admit_time.apply(lambda n: n/3600)
    return df[numerical], df[categorical].applymap(int)

def _int64_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))
def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

@pu.memoize('cache/_split_dataframe_{0:d}_idempotent.pkl')
def split_dataframe(n_frequent):
    usecols, dtype, fillna, numerical_headers, categorical_headers, treatments_headers = \
        get_frequent_headers(n_frequent=n_frequent)
    mimic = parse_csv(usecols, dtype)
    static_data_numerical, static_data_categorical = get_static_data()

    pu.dump({'numerical_ts': len(numerical_headers),
             'categorical_ts': len(categorical_headers),
             'treatments_ts': len(treatments_headers),
             'numerical_static': static_data_numerical.shape[1],
             'categorical_static': static_data_categorical.shape[1],
        }, 'dataset/feature_numbers.pkl.gz')

    print("Computing icustay indices...")
    valid_icustays = list(set(mimic.index))
    print("Splitting...")
    step = int(math.ceil(len(valid_icustays) / N_PROCESSORS))
    j = 0
    for i in range(0, len(valid_icustays), step):
        pu.dump(mimic.loc[valid_icustays[i:i+step]].fillna(fillna),
                'dataset/dataset_{:d}.pkl.gz'.format(j))
        j += 1


def write_tfrecords(mimic, i, numerical_headers, categorical_headers, treatments_headers,
          static_data_numerical, static_data_categorical):
    with tf.python_io.TFRecordWriter("cache/hv_{:d}.tfrecords".format(i)) as writer:
        n_examples = 0
        for icustay_id, df in mimic.groupby(level=0):
            print("Doing icustay_id", icustay_id, "...")
            ventilation_ends = df[['hour', 'B pred last_ventilator']].dropna().values
            if len(ventilation_ends) == 0:
                continue
            ts_len = ventilation_ends[-1,0]+1
            if ts_len == 0:
                print("LENGTH 0: icustay_id", icustay_id)
                continue

            time_until_label = np.empty([ts_len], dtype=np.float32)
            for hour, _ in reversed(ventilation_ends):
                time_until_label[:hour+1] = np.arange(hour, -1, -1, dtype=np.float32)

            # Label is 1 when the patient will be re-ventilated
            label = np.zeros([ts_len], dtype=np.bool)
            if len(ventilation_ends) > 1:
                label[:ventilation_ends[-2,0]+1] = 1

            hours = df['hour'].values

            numerical_ts = np.empty([ts_len, len(numerical_headers)], dtype=np.float32)
            numerical_ts[...] = np.nan
            numerical_ts_dt = np.zeros([len(numerical_headers)], dtype=np.float32) + 1000
            numerical_arr = df[numerical_headers].values

            # categorical_ts starts with being all -1
            categorical_ts = -np.ones([ts_len, len(categorical_headers)], dtype=np.int8)
            categorical_ts_dt = np.zeros([len(categorical_headers)], dtype=np.float32) + 1000
            categorical_df = df[categorical_headers]
            categorical_df_usable = ~(categorical_df.isnull().values)


            treatments_ts = np.zeros([ts_len, len(treatments_headers)], dtype=np.bool)
            treatments_df = df[treatments_headers].values

            i=0
            while i < len(hours) and hours[i] < ts_len:
                overwrite = ~np.isnan(numerical_arr[i])
                cdu = categorical_df_usable[i]
                if hours[i] <= 0:
                    # make it positive, hours[i] should contain non-positive values
                    numerical_ts_dt[overwrite] = -hours[i]
                    categorical_ts_dt[cdu] = -hours[i]
                else:
                    treatments_ts[hours[i]] = treatments_df[i]

                numerical_ts[max(0, hours[i]), overwrite] = numerical_arr[i, overwrite]
                categorical_ts[max(0, hours[i]), cdu] = (
                    categorical_df.iloc[i, cdu].apply(int).values)

                i += 1

            numerical_static = static_data_numerical.loc[icustay_id].values
            assert len(numerical_static.shape) == 1
            categorical_static = static_data_categorical.loc[icustay_id].values
            assert len(categorical_static.shape) == 1

            data = {
                'icustay_id': [icustay_id],
                'numerical_static': numerical_static,
                'categorical_static': categorical_static,
                'numerical_ts_dt': numerical_ts_dt,
                'categorical_ts_dt': categorical_ts_dt,

                'time_until_label': time_until_label,
                'label': label.astype(np.float32),
                'numerical_ts': numerical_ts,
                'categorical_ts': categorical_ts,
                'treatments_ts': treatments_ts.astype(np.float32),
                'ventilation_ends': ventilation_ends[:,0],
            }
            example = example_from_data(data)
            writer.write(example.SerializeToString())

            if n_examples > 10:
                break

@pu.memoize('dataset/number_of_categories_{0:d}.pkl.gz', log_level='warn')
def compute_number_of_categories(n_frequent):
    _, static_data_categorical = get_static_data()

    usecols, dtype, _, _, categorical_headers, _ = \
        get_frequent_headers(n_frequent=n_frequent)
    mimic = parse_csv(usecols, dtype)

    out = {}
    l = []
    for h in categorical_headers:
        l.append(len(mimic[h].cat.categories))
    out['categorical_ts'] = l
    out['categorical_static'] = (
        (static_data_categorical.max(axis=0).values+1).tolist())
    return out

@pu.memoize('cache/_training_split_idempotent_{0:d}.pkl', log_level='warn')
def join_tfrecords_training_test_vali(n_output_folds):
    records = list(it.chain(
        *(tf.python_io.tf_record_iterator('cache/hv_{:d}.tfrecords'.format(i))
                for i in range(N_PROCESSORS))))
    if not os.path.isdir('dataset'):
        os.mkdir('dataset')
    def write(data, fname):
        p = os.path.join('dataset', fname)
        with tf.python_io.TFRecordWriter(p) as f:
            for d in data:
                f.write(d)

    print("We have {:d} records in total".format(len(records)))
    np.random.shuffle(records)
    n_testing = len(records) // 5
    print("20% ({:d}) records for testing".format(n_testing))
    write(records[:n_testing], 'test.tfrecords')

    records = records[n_testing:]
    n_folds = 5
    n_validation = int(math.ceil((len(records)-n_testing) / n_folds))
    print("16% ({:d}) records for validation".format(n_validation))
    print("And the rest ({:d}) for training"
            .format(len(records)-n_validation))

    for j in range(n_output_folds):
        i = j*n_validation
        write(it.chain(records[:i], records[i+n_validation:]),
                'train_{:d}.tfrecords'.format(j))
        write(records[i:i+n_validation],
                'validation_{:d}.tfrecords'.format(j))

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'number_of_categories':
        compute_number_of_categories(int(sys.argv[2]))
    elif sys.argv[1] == 'join_tfrecords_training_test_vali':
        join_tfrecords_training_test_vali(int(sys.argv[2]))
    elif sys.argv[1] == 'split_dataframe':
        split_dataframe(int(sys.argv[2]))
    elif sys.argv[1] == 'write_tfrecords':
        _, _, _, numerical_headers, categorical_headers, treatments_headers = \
            get_frequent_headers(n_frequent=int(sys.argv[2]))
        del _
        i = int(sys.argv[3])
        mimic = pu.load('dataset/dataset_{:d}.pkl.gz'.format(i))
        static_data_numerical, static_data_categorical = get_static_data()
        write_tfrecords(mimic, i, numerical_headers, categorical_headers, treatments_headers,
            static_data_numerical, static_data_categorical)
    else:
        raise ValueError(sys.argv[1])
