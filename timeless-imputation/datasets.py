from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()

import numpy as np
import pickle_utils as pu
import pandas as pd
import collections

py2ri = rpy2.robjects.pandas2ri.py2ri
ri2py = rpy2.robjects.pandas2ri.ri2py
R = rpy2.robjects.r

base = importr("base")
R_utils = importr("utils")
mlbench = importr("mlbench")

import utils
import os

__all__ = ['datasets', 'benchmark', 'memoize']

def datasets():
    columns_excluded = {
        "Ionosphere": ["V2", "Class"],
        "BostonHousing": ["medv", "chas"],
        #"BreastCancer": ["Class"],
        #"DNA": ["Class"],
    }
    dfs = {}
    for name, excluded in columns_excluded.items():

        R_utils.data(name)
        dfs[name] = (ri2py(base.as_data_frame(R[name]))
                     .drop(excluded, axis=1))
        if name == "Ionosphere":
            dfs[name].V1 = dfs[name].V1.astype(np.float)
    return dfs

def memoize(f):
    @pu.memoize("{0:s}.pkl.gz")
    def memoized_f(_path, *args, **kwargs):
        return f(*args, **kwargs)
    return memoized_f

def benchmark(impute_methods, datasets, path="impute_benchmark"):
    if not os.path.exists(path):
        os.mkdir(path)

    def recursive_defaultdict():
        return collections.defaultdict(recursive_defaultdict, {})
    table = recursive_defaultdict()

    def do_mcar_rows(dataset_, proportion):
        return utils.mcar_rows(dataset_, proportion**.5, proportion**.5)
    for algo_name, impute_f in impute_methods.items():
        for data_name, full_data in datasets.items():
            for ampute_fun, ampute_fun_name in [
                    (memoize(utils.mcar_total), 'MCAR_total'),
                    (memoize(do_mcar_rows), 'MCAR_rows')]:
                for proportion in [.1, .3, .5, .7, .9]:
                    amputed_name = '{:s}_{:s}_{:.1f}'.format(
                            data_name, ampute_fun_name, proportion)
                    amputed_data = ampute_fun(
                        os.path.join(path, 'amputed_'+amputed_name),
                        full_data,
                        proportion)
                    imputed_data = impute_f(
                        os.path.join(path, 'imputed_{:s}_{:s}'.format(
                            algo_name, amputed_name)),
                        amputed_data)

                    table[algo_name]['RMSE_best'][data_name][ampute_fun_name]\
                        [str(proportion)] = utils.mean_rmse(
                            np.isnan(amputed_data.values),
                            full_data.values,
                            list(d.values for d in imputed_data))

                    if len(imputed_data) > 1:
                        table[algo_name]['RMSE'][data_name][ampute_fun_name]\
                            [str(proportion)] = utils.rmse(
                                np.isnan(amputed_data.values),
                                full_data.values,
                                list(d.values for d in imputed_data))

    cols_1 = []
    cols_2 = []
    rows_1 = None
    rows_2 = None
    rows_3 = None
    for algo in sorted(table.keys()):
        cols_l = []
        for measurement in sorted(table[algo].keys()):
            cols_l.append(measurement)
            _rows_1 = []
            _rows_2 = []
            _rows_3 = []
            for data in sorted(table[algo][measurement]):
                ampute_l = []
                for ampute in sorted(table[algo][measurement][data]):
                    proportion_l = list(sorted(table[algo][measurement][data][ampute].keys()))
                    _rows_3 += proportion_l
                    ampute_l += [ampute]*len(proportion_l)
                _rows_2 += ampute_l
                _rows_1 += [data]*len(ampute_l)
            if rows_1 is None:
                rows_1 = _rows_1
                rows_2 = _rows_2
                rows_3 = _rows_3
            else:
                assert rows_1 == _rows_1
                assert rows_2 == _rows_2
                assert rows_3 == _rows_3


        cols_1 += [algo]*len(cols_l)
        cols_2 += cols_l

    data = -np.ones(shape=[len(rows_1), len(cols_1)], dtype=np.float)
    for i, (r1, r2, r3) in enumerate(zip(rows_1, rows_2, rows_3)):
        for j, (c1, c2) in enumerate(zip(cols_1, cols_2)):
            data[i,j] = table[c1][c2][r1][r2][r3]

    return pd.DataFrame(data,
                        index=[np.array(a) for a in [rows_1, rows_2, rows_3]],
                        columns=[np.array(a) for a in [cols_1, cols_2]])

if __name__ == '__main__':
    print(benchmark({'MICE': utils.impute_mice, 'MissForest': utils.impute_missforest}, datasets()))
