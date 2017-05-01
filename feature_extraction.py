import numpy as np
import pandas as pd
import tsfresh
import csv
import gzip
import os
import pickle
import itertools as it
import collections
import multiprocessing
import tsfresh

def get_headers(table):
    with gzip.open('../mimic-clean/{:s}.csv.gz'.format(table), 'rt', newline='') as csvf:
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

def read_event_data(fname, dtype):
    if os.path.isdir(fname):
        l = os.listdir(fname)
        return (read_event_data(f) for f in filter(
            lambda n: n[0] != '.' and n[-4:] == '.pkl', l))

    def df_with_name(k, contents):
        return pd.DataFrame(np.array(contents,
            dtype=[("id", np.int32),
                   ("minute", np.int32),
                   (k, dtype[k])]))

    with open(fname, 'rb') as f:
        evs = {}
        d = pickle.load(f)
        evs_labels = d['labels']
        del d['labels']
        headers = list(filter(lambda k: dtype[k] != 'category', d.keys()))
        for k in headers:
            evs[k] = df_with_name(k, contents=sorted(d[k]))
        del d
    prev_icustay_id = None
    labels_series = []
    data_series = dict(map(lambda k: (k, df_with_name(k, [])), headers))

    for icustay_id, end_time, *labels in evs_labels:
        if icustay_id == prev_icustay_id:
            n_vent += 1
        else:
            n_vent = 0
            prev_icustay_id = icustay_id
        for i, hours_before in enumerate([4, 8, 12, 24]):
            id = icustay_id*1000 + n_vent*10 + i
            labels_series.append((id,)+tuple(labels))
            for k, df in evs.items():
                df = df[df.id == id]
                df = df[df.minute <= end_time - hours_before*60]
                prev_len = data_series[k].shape[0]
                data_series[k] = pd.concat([data_series[k], df])
                data_series[k].iloc[prev_len:, 0] = id
    return data_series, labels_series

if __name__ == '__main__':
    zero_headers = get_headers('outputevents')
    bool_headers = (get_headers('procedureevents_mv') +
                    get_headers('drugevents'))
    nan_headers = get_headers('labevents') + get_headers('chartevents')

    dtype = dict(it.chain(
        map(lambda e: (e, determine_type(e, True)), nan_headers + zero_headers),
        map(lambda e: (e, determine_type(e, False)), bool_headers),
        map(lambda e: (e, np.int32), ["id", "minute"])))
    data, labels = read_event_data('../mimic-clean/no_nan/data_lt_276740.pkl', dtype)
    import code; code.interact(local=locals())
