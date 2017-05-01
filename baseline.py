#!/usr/bin/env python3

import psycopg2
import pickle
import nltk
import urllib
import os
import numpy as np
import pandas as pd
import collections
import zipfile
import itertools as it
from memoize_pickle import memoize_pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn_pandas import DataFrameMapper

class BoW:
    def __init__(self, dims=100):
        GLOVE_FILE = 'glove.6B.{:d}d.txt'.format(dims)
        GLOVE_ZIP = '../glove.6B.zip'
        self.dims = dims
        if not os.path.isfile(GLOVE_ZIP):
            print("Downloading GloVe...")
            urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip",
                                       filename=GLOVE_ZIP)
        self.vocab = {}
        with zipfile.ZipFile(GLOVE_ZIP, 'r') as z:
            with z.open(GLOVE_FILE, 'r') as f:
                for line in f.readlines():
                    l = line.split()
                    w = str(l[0], 'utf8')
                    self.vocab[w] = np.asarray(list(float(d) for d in l[1:]), dtype=np.float32)
        self.tt = nltk.TweetTokenizer()
        self.token_counts = collections.defaultdict(lambda: 0, {})

    def bow(self, diagnosis):
        s = np.zeros(self.dims, dtype=np.float32)
        n = 0
        for token in self.tt.tokenize(diagnosis):
            token = token.lower()
            self.token_counts[token] += 1
            if token in self.vocab:
                n += 1
                s += self.vocab[token]
        if n != 0:
            s /= n
        return s

@memoize_pickle("static_data.pkl")
def fetch_data(add_diagnosis_bow=True):
    conn_string = "host='localhost' dbname='adria' user='adria' password='adria'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    headers = ["icustay_id", "r_admit_time", "c_admit_type",
              "c_admit_location", "c_insurance", "c_marital_status",
              "c_ethnicity", "b_gender", "r_age", "i_previous_admissions",
              "i_previous_icustays", "r_pred_discharge_time",
              "c_pred_discharge_location", "r_pred_death_time",
              "b_pred_died_in_hospital", "info_diagnosis"]
    cursor.execute("SET search_path TO mimiciii")
    cursor.execute(("SELECT {:s} FROM static_icustays "
                    "WHERE subject_id IN "
                    "(SELECT * FROM metavision_patients)")
                    .format(", ".join(headers)))
    np_t = {'r': np.float32, 'c': np.int32, 'i': np.int32, 'b': np.bool}
    df_t = {'r': np.float32, 'c': 'category', 'i': np.int32, 'b': np.bool}
    rows = cursor.fetchall()

    # Embed diagnoses
    # icustay_id goes first but it already starts with i
    np_types = list((h, np_t[h[0]]) for h in headers[1:-1])
    df_types = dict((h, np_t[h[0]]) for h in headers[1:-1])
    arr = np.array(list(r[1:-1] for r in rows), dtype=np_types)
    df = pd.DataFrame(arr, index=list(r[0] for r in rows))
    df = df.fillna({'r_pred_death_time': 80.*365*24}) # 80 years

    if add_diagnosis_bow:
        bow = BoW()
        l = np.array(list(bow.bow(r[-1]) for r in rows), dtype=np.float32)
        bow_df = pd.DataFrame(l, index=df.index, columns=list(
            'r_diagnosis_{:d}'.format(i) for i in range(l.shape[1])))
        df = df.join(bow_df)
    msk = np.random.rand(len(df)) < 0.8
    return df[msk], df[~msk]

def convert_to_sklearn(df):
    def processing_steps(header):
        if header[0] == 'c':
            return header, preprocessing.LabelBinarizer()
        elif header[:6] == 'r_diag':
            return header, None
        return [header], preprocessing.RobustScaler()
    return (DataFrameMapper(list(map(processing_steps, df.keys())))
            .fit_transform(df.copy()))

ALL_LABELS = ["c_pred_discharge_location", "r_pred_discharge_time",
                "r_pred_death_time", "b_pred_died_in_hospital"]

if __name__ == '__main__':
    train, test = fetch_data()
    label = ALL_LABELS[0]
    X = convert_to_sklearn(train.drop(ALL_LABELS, axis=1))
    y = convert_to_sklearn(train[[label]])
    pipeline = Pipeline([('forest', RandomForestClassifier())])
    print(cross_val_score(pipeline, X, y, n_jobs=-1))
