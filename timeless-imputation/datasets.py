from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
import rpy2.rinterface
import numpy as np
import pickle_utils as pu
import pandas as pd
import collections
import utils
import os
import collections
import unittest
import sys

rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()
py2ri = rpy2.robjects.pandas2ri.py2ri
ri2py = rpy2.robjects.pandas2ri.ri2py
R = rpy2.robjects.r

base = importr("base")
R_utils = importr("utils")
mlbench = importr("mlbench")

__all__ = ['datasets', 'benchmark', 'memoize']

dataframe_like = utils.dataframe_like

def datasets(exclude_labels=True):
    columns_excluded = {
        "Ionosphere": (["Class"] if exclude_labels else []),
        "BostonHousing": (["medv"] if exclude_labels else []),
        "BreastCancer": ["Id"] + (["Class"] if exclude_labels else []),
        # "DNA": ["Class"], # Class
        "Soybean": (["Class"] if exclude_labels else []),
        "Servo": (["Class"] if exclude_labels else []),
        "LetterRecognition": (["lettr"] if exclude_labels else []),
        "Shuttle": (["Class"] if exclude_labels else []),
    }
    dfs = {}
    for name, excluded in columns_excluded.items():
        R("data({:s})".format(name))
        R("df <- as.data.frame({:s})".format(name))
        print("+++ Importing", name)

        df, categories = utils.df_from_R("df", more_than_one_value=True)
        df = df.drop(excluded, axis=1)
        for e in excluded:
            if e in categories:
                categories.remove(e)
        dfs[name] = df, categories
    return dfs


def memoize(f):
    @pu.memoize("{0:s}.pkl.gz")
    def memoized_f(_path, *args, **kwargs):
        return f(*args, **kwargs)
    return memoized_f


def benchmark(impute_methods, datasets, do_not_compute=False,
              path="impute_benchmark"):
    if not os.path.exists(path):
        os.mkdir(path)

    def recursive_defaultdict():
        return collections.defaultdict(recursive_defaultdict, {})
    table = recursive_defaultdict()

    def do_mcar_rows(dataset_, proportion):
        return utils.mcar_rows(dataset_, proportion**.5, proportion**.5)

    tests_to_perform = []
    for b in datasets.items():
        for c in [(memoize(utils.mcar_total), 'MCAR_total'),
                    (memoize(do_mcar_rows), 'MCAR_rows')]:
            for d in [.1, .3, .5, .7, .9]:
                for e in ['mean_std']:
                    tests_to_perform.append((b, c, d, e))
    del b, c, d, e

    np.random.shuffle(tests_to_perform)
    #tests_to_perform = [(("Ionosphere", datasets["Ionosphere"]),
    #                     (memoize(utils.mcar_rows), 'MCAR_rows'),
    #                     .3, 'mean_std')]

    for ((data_name, (full_data, cat_keys)), (ampute_fun, ampute_fun_name),
         proportion, norm_type) in tests_to_perform:
        for (algo_name, impute_f) in impute_methods.items():
            amputed_name = '{:s}_{:s}_{:.1f}'.format(
                    data_name, ampute_fun_name, proportion)
            amputed_data = ampute_fun(
                os.path.join(path, 'amputed_'+amputed_name),
                full_data,
                proportion)
            _data, moments = utils.normalise_dataframes(
                amputed_data, full_data, method=norm_type)
            _ad, _fd = _data

            if (algo_name == 'MissForest' and proportion == .9 and
                data_name == "Soybean" and
                ampute_fun_name == 'MCAR_total') or (
                    do_not_compute and not
                    os.path.exists(os.path.join(
                        path, 'imputed_{:s}_{:s}.pkl.gz'.format(
                            algo_name, amputed_name)))
                    and not os.path.exists(os.path.join(
                        path, 'imputed_{:s}_{:s}/checkpoint'.format(
                            algo_name, amputed_name)))
                    and not os.path.exists(os.path.join(
                        path, 'imputed_{:s}_{:s}/mf_out.pkl.gz'.format(
                            algo_name, amputed_name)))
                    and not os.path.exists(os.path.join(
                        path, 'imputed_{:s}_{:s}/params.pkl.gz'.format(
                            algo_name, amputed_name)))):
                table['RMSE'][algo_name][norm_type][data_name]\
                    [ampute_fun_name][str(proportion)] = np.nan
                table['NRMSE'][algo_name][norm_type][data_name]\
                    [ampute_fun_name][str(proportion)] = np.nan
                table['PFC'][algo_name][norm_type][data_name]\
                    [ampute_fun_name][str(proportion)] = np.nan
                continue
            else:
                run_name = 'imputed_{:s}_{:s}'.format(algo_name, amputed_name)
                if not do_not_compute:
                    print("Computing ", run_name)
                _id = impute_f(
                    os.path.join(path, run_name), (_ad, cat_keys),
                    full_data=(_fd, cat_keys))

            imputed_data = utils.unnormalise_dataframes(moments, _id)
            d = utils.reconstruction_metrics(amputed_data, full_data,
                                             imputed_data)
            if not do_not_compute:
                print("JUST GOT SOME RESULTS FOR:", algo_name)
            for k, v in d.items():
                table[k][algo_name][norm_type][data_name]\
                    [ampute_fun_name][str(proportion)] = v
                if not do_not_compute:
                    print(k, v)


    def get_multiindex(d, levels=3):
        if levels == 1:
            return [sorted(list(d.keys()))]
        index = list([] for _ in range(levels))
        for key in sorted(d.keys()):
            item = d[key]
            lower_levels = get_multiindex(item, levels=levels-1)
            for i in range(levels-1):
                index[i+1] += lower_levels[i]
            index[0] += [key]*len(lower_levels[0])
        return index

    cols = get_multiindex(table, levels=2)
    rows = None
    for algo, measurement in zip(*cols):
        _rows = get_multiindex(table[algo][measurement], levels=4)
        if rows is None:
            rows = _rows
        assert rows == _rows, "All rows of the passed dict must be the same"

    data = np.empty(shape=[len(rows[0]), len(cols[0])], dtype=np.float)
    data[...] = np.inf  # Detect errors when looking at the table
    for i, (r1, r2, r3, r4) in enumerate(zip(*rows)):
        for j, (c1, c2) in enumerate(zip(*cols)):
            data[i,j] = table[c1][c2][r1][r2][r3][r4]

    return pd.DataFrame(data,
                        index=[np.array(a) for a in rows],
                        columns=[np.array(a) for a in cols])
if __name__ == '__main__':
    dsets = {"BostonHousing": datasets()["BostonHousing"]}
    baseline = benchmark({
        'MissForest': memoize(utils.impute_missforest),
    }, dsets, do_not_compute=False)
    print(baseline)
