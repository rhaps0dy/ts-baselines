#!/usr/bin/env python3

import psycopg2
import pickle
import nltk
import urllib
import os
import collections
import zipfile
import itertools as it
from memoize_pickle import memoize_pickle

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from sklearn_pandas import DataFrameMapper

## DeprecationWarning with DataFrameMapper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class BoW:
    def __init__(self, data, bow_type, dims=100):
        self.dims = dims
        self.tt = nltk.TweetTokenizer()
        if bow_type == 'glove':
            print("type glove")
            GLOVE_FILE = 'glove.6B.{:d}d.txt'.format(dims)
            GLOVE_ZIP = '../glove.6B.zip'
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
            self.bow = self._glove_bow
        elif bow_type == "count":
            vocab_counts = collections.Counter()
            for d in data:
                vocab_counts.update(e.lower() for e in self.tt.tokenize(d))
            vocab, counts = zip(*vocab_counts.most_common(dims))
            self.vocab = dict(map(lambda t: (t[1], t[0]), enumerate(vocab)))
            self.bow = self._count_bow
        else:
            raise ValueError("Unknown bow_type: {:s}".format(bow_type))

    def _glove_bow(self, diagnosis):
        s = np.zeros(self.dims, dtype=np.float32)
        n = 0
        for token in self.tt.tokenize(diagnosis):
            token = token.lower()
            if token in self.vocab:
                n += 1
                s += self.vocab[token]
        if n != 0:
            s /= n
        return s

    def _count_bow(self, diagnosis):
        s = np.zeros(self.dims, dtype=np.float32)
        for token in self.tt.tokenize(diagnosis):
            if token in self.vocab:
                s[self.vocab[token]] += 1
        return s

@memoize_pickle("ventilation_df.pkl")
def ventilation_df():
    ventilations = []
    data_dir = '../mimic-clean/no_nan'
    for fname in os.listdir(data_dir):
        if fname[-4:] != ".pkl":
            continue
        with open(os.path.join(data_dir, fname), 'rb') as f:
            ventilations += pickle.load(f)["labels"]
    ventilations.sort()
    d = {}
    index, d["r_start_minute"], d["b_pred_vent_last"], _death_time = (
        zip(*ventilations))
    d["i_previous_ventilations"] = []
    prev_icustay_id = None
    for icustay_id in index:
        if prev_icustay_id != icustay_id:
            prev_icustay_id = icustay_id
            pv = 0
        else:
            pv += 1
        d["i_previous_ventilations"].append(pv)
    return pd.DataFrame(d, index=index)

def fetch_data(add_diagnosis_bow):
    @memoize_pickle("static_data_{:s}.pkl"
            .format(add_diagnosis_bow))
    def f():
        return _fetch_data(add_diagnosis_bow)
    return f()

def _fetch_data(add_diagnosis_bow, train_proportion=0.8):
    conn_string = "host='localhost' dbname='adria' user='adria' password='adria'"
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    headers = ["icustay_id", "r_admit_time", "c_admit_type",
              "c_admit_location", "c_insurance", "c_marital_status",
              "c_ethnicity", "b_gender", "r_age", "i_previous_admissions",
              "i_previous_icustays",
              "r_pred_discharge_time", "c_pred_discharge_location",
              "r_pred_death_time", "b_pred_died_in_hospital",
               "info_diagnosis"]
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
    if 'r_pred_death_time' in df:
        df = df.fillna({'r_pred_death_time': 80.*365*24}) # 80 years

    if add_diagnosis_bow != "nobow":
        bow = BoW(data=(r[-1] for r in rows), bow_type=add_diagnosis_bow)
        l = np.array(list(bow.bow(r[-1]) for r in rows), dtype=np.float32)
        bow_df = pd.DataFrame(l, index=df.index, columns=list(
            'r_diagnosis_{:d}'.format(i) for i in range(l.shape[1])))
        df = df.join(bow_df)

    ventilations = ventilation_df()
    df = df.join(ventilations)

    msk = np.random.rand(len(df)) < train_proportion
    return df[msk], df[~msk]

def convert_to_sklearn(df):
    def processing_steps(header):
        if header[0] == 'c':
            return header, preprocessing.LabelBinarizer()
        elif header[:6] == 'r_diag':
            return header, None
        return [header], preprocessing.RobustScaler()
    return DataFrameMapper(list(map(processing_steps, df.keys())))

def geom(scale):
    return scipy.stats.geom(p=1-np.exp(-1/scale))

def test_classifier(data_X, data_y, p_after_mapper, param_distributions, name):
    pipeline = Pipeline([
        ('mapper', convert_to_sklearn(data_X)),
    ] + p_after_mapper)

    random_search = RandomizedSearchCV(pipeline,
            param_distributions=param_distributions,
            n_iter=10,
            scoring='roc_auc',
            cv=5,
            n_jobs=-1)
    best_pipeline = random_search.fit(data_X.ix['train'], data_y.ix['train'].values)
    df = pd.DataFrame(best_pipeline.cv_results_)
    results = df[df.rank_test_score==1][
        ['mean_test_score', 'mean_train_score']].values
    test_score = best_pipeline.score(data_X.ix['test'], data_y.ix['test'])
    print(name, end='. ')
    print("AUC. Test: {:.4f}, Validation: {:.4f}, Training: {:.4f}"
            .format(test_score, *results[0]))
    print("Elapsed time: {:.2f}s".format(
        df[['mean_fit_time', 'mean_score_time']].values.sum() *
        best_pipeline.n_splits_))
    print(best_pipeline.best_params_)

if __name__ == '__main__':
    pd.set_option('display.max_columns', 7)
    for data_kind in "nobow", "glove", "count":
        data = pd.concat(list(fetch_data(data_kind)), keys=["train", "test"])
        # Drop patients that have never been ventilated
        data = data.dropna().astype({"b_pred_vent_last": bool})
        data_X = data.filter(regex=r'^..(?!pred).*$')
        data_y = data["b_pred_vent_last"]

        test_classifier(data_X, data_y,
                [('forest', RandomForestClassifier())],
                param_distributions = {
                    'forest__n_estimators': geom(30),
                    'forest__criterion': ['gini', 'entropy'],
                },
                name ="Random Forest Classifier ({:s})".format(data_kind))
