import numpy as np
import pandas as pd
import tsfresh
import csv
import gzip
import itertools as it
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

if __name__ == '__main__':
    zero_headers = get_headers('outputevents')
    bool_headers = (get_headers('procedureevents_mv') +
                    get_headers('drugevents'))
    nan_headers = get_headers('labevents') + get_headers('chartevents')

    dtype = dict(it.chain(
        map(lambda e: (e, determine_type(e, True)), nan_headers + zero_headers),
        map(lambda e: (e, determine_type(e, False)), bool_headers),
        map(lambda e: (e, np.int32), ["icustay_id", "hour"])))

    n_headers = len(zero_headers)+len(bool_headers)+len(nan_headers)+2
    usecols = list(range(1, 1+n_headers)) # ignore "subject_id"

    fillna = dict(it.chain(
        map(lambda e: (e, 0.0), zero_headers),
        map(lambda e: (e, False),
            list(filter(lambda h: h[1:7] != ' pred ', bool_headers))),
#        map(lambda e: (e, 'bfill'),
#            list(filter(lambda h: h[1:7] == ' pred ', bool_headers))),
    ))

    df = pd.read_csv('../mimic.csv.gz', nrows=10000,
                     index_col=False, engine='c',
                     true_values=[b'1'], false_values=[b'0'],
                     usecols=usecols, dtype=dtype).fillna(fillna)

    observations = df.filter(regex=r'^.(?! pred ).*$').fillna(method='ffill').fillna(method='bfill').dropna(axis='columns')
    labels = df.filter(regex=r'^. pred .*$')
    ventilation_ends = np.argwhere(pd.notnull(labels.iloc[:,0]))

    icustays_start = {}
    for i, v in enumerate(observations['icustay_id']):
        if v not in icustays_start:
            icustays_start[v] = i
    start_icustays = sorted(list(map(lambda t: (t[1],t[0]), icustays_start.items())))
    labels_notime = []
    icustay_labels = collections.defaultdict(lambda: [], {})
    sicu_i = -1
    prev_icustay = None
    for v_end in ventilation_ends:
        while start_icustays[sicu_i+1][0] <= v_end:
            sicu_i += 1
        icustay_id = start_icustays[sicu_i][1]
        if icustay_id != prev_icustay:
            prev_icustay = icustay_id
            n_vent = -1
        n_vent += 1
        for hours_before in 4, 8, 12, 24:
            id = icustay_id*1000 + n_vent*10 + hours_before
            icustay_labels[icustay_id].append(id)
            labels_notime.append((id,) + tuple(ventilation_labels.iloc[v_end,:]))



    tsfresh.feature_selection.FeatureSignificanceTestsSettings.n_processes = 8
    tsfresh.feature_extraction.FeatureExtractionSettings.n_processes = 8
    #X = tsfresh.extract_relevant_features(observations, labels, column_id='icustay_id', column_sort='hour')

