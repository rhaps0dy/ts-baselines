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

rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()
py2ri = rpy2.robjects.pandas2ri.py2ri
ri2py = rpy2.robjects.pandas2ri.ri2py
R = rpy2.robjects.r

base = importr("base")
R_utils = importr("utils")
mlbench = importr("mlbench")

__all__ = ['datasets', 'benchmark', 'memoize']


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

def percentage_falsely_classified(amputed_data, full_data, imputed_data):
    """Percentage falsely classified. It only measures the performance on
    entries that are present in the `full_data`"""
    def key_filter(f):
        df = full_data
        return list(filter(lambda k: f(df[k].dtype), df.keys()))

    def count_wrong_classifications(truth, attempts):
        attempts = np.stack(attempts)
        assert truth.shape[0] == attempts.shape[1]
        assert len(truth.shape) == 1
        wrong = 0
        for t, a in zip(iter(truth), iter(attempts.T)):
            wrong += (t != collections.Counter(a)
                        .most_common(1)[0][0])
        return wrong, len(truth)

    int_keys = key_filter(lambda dt: dt == np.int32)
    int_mask = (amputed_data[int_keys].values == utils.NA_int32)
    int_mask &= (full_data[int_keys].values != utils.NA_int32)
    wrong, total = count_wrong_classifications(
        full_data[int_keys].values[int_mask],
        list(i[int_keys].values[int_mask] for i in imputed_data))

    return wrong, total


def benchmark(impute_methods, datasets, do_not_compute=False,
              path="impute_benchmark"):
    if not os.path.exists(path):
        os.mkdir(path)

    def recursive_defaultdict():
        return collections.defaultdict(recursive_defaultdict, {})
    table = recursive_defaultdict()

    def do_mcar_rows(dataset_, proportion):
        return utils.mcar_rows(dataset_, proportion**.5, proportion**.5)
    for algo_name, impute_f in impute_methods.items():
        for data_name, (full_data, cat_keys) in datasets.items():
            for ampute_fun, ampute_fun_name in [
                    (memoize(utils.mcar_total), 'MCAR_total'),
                    (memoize(do_mcar_rows), 'MCAR_rows')]:
                for proportion in [.1, .3, .5, .7, .9]:
                    for norm_type in ['mean_std']:  # 'min_max']:
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
                                       algo_name, amputed_name)))):
                            table['RMSE'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = np.nan
                            table['NRMSE'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = np.nan
                            table['total_cats'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = np.nan
                            table['PFC'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = np.nan
                            continue
                        else:
                            try:
                                _id = impute_f(
                                    os.path.join(path, 'imputed_{:s}_{:s}'.format(
                                        algo_name, amputed_name)), (_ad, cat_keys),
                                    full_data=(_fd, cat_keys))
                            except Exception as e:
                                if do_not_compute:
                                    table['RMSE'][algo_name][norm_type][data_name]\
                                        [ampute_fun_name][str(proportion)] = np.nan
                                    table['NRMSE'][algo_name][norm_type][data_name]\
                                        [ampute_fun_name][str(proportion)] = np.nan
                                    table['total_cats'][algo_name][norm_type][data_name]\
                                        [ampute_fun_name][str(proportion)] = np.nan
                                    table['PFC'][algo_name][norm_type][data_name]\
                                        [ampute_fun_name][str(proportion)] = np.nan
                                    continue
                                else:
                                    raise e


                        imputed_data = utils.unnormalise_dataframes(moments, _id)
                        # Normalised RMSE:
                        # For normalised RMSE, we take mean_std normalisation
                        # over the missing values only
                        numerical_keys = list(moments[0].keys())
                        if len(numerical_keys) == 0:
                            table['RMSE'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = np.nan
                            table['NRMSE'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = np.nan
                        else:
                            adv = amputed_data[numerical_keys].values
                            missing_mask = np.isnan(adv)
                            idv = list(df[numerical_keys].values[missing_mask]
                                    for df in imputed_data)
                            mean_idv = np.mean(idv, axis=0)
                            fdv = full_data[numerical_keys].values[missing_mask]
                            assert fdv.shape == mean_idv.shape
                            nrmse = (np.mean((fdv-mean_idv)**2) / np.var(fdv))**.5
                            assert len(nrmse.shape) == 0

                            table['NRMSE'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = nrmse

                            # RMSE of 0-1 normalised data
                            rmse_fd, rmse_ad, *rmse_id = utils.normalise_dataframes(
                                full_data[numerical_keys],
                                amputed_data[numerical_keys],
                                *(i[numerical_keys] for i in imputed_data),
                                method='min_max')[0]

                            table['RMSE'][algo_name][norm_type][data_name]\
                                [ampute_fun_name][str(proportion)] = utils.mean_rmse(
                                    np.isnan(rmse_ad.values), rmse_fd.values,
                                    list(d.values for d in rmse_id))

                        wrong, total = percentage_falsely_classified(
                            amputed_data, full_data, imputed_data)
                        table['total_cats'][algo_name][norm_type][data_name]\
                            [ampute_fun_name][str(proportion)] = total
                        table['PFC'][algo_name][norm_type][data_name]\
                            [ampute_fun_name][str(proportion)] = wrong/total

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


def dataframe_like(dataframe, new_values):
    return pd.DataFrame(new_values, index=dataframe.index,
                        columns=dataframe.columns)


if __name__ == '__main__':
    import denoising_ae
    import tensorflow as tf
    np.seterr(all='raise')
    dsets = datasets()
    #baseline = benchmark({'MICE': memoize(utils.impute_mice),
    #                      'MissForest': memoize(utils.impute_missforest)}, dsets)
    dsets = dict(filter(lambda k: k[0] == 'LetterRecognition', dsets.items()))
    benchmark({'MLP_2x256_SGD': lambda log_path, dataset, full_data: denoising_ae.impute(
        log_path, dataset, full_data,
        number_imputations=64,
        number_layers=2,
        hidden_units=256,
        seed=0,
        init_type=denoising_ae.SELU_initializer,
        batch_size=256,
        nlin=tf.nn.tanh,
        residual=False,
        patience=10,
        learning_rate=1e-4,
        corruption_prob=0.5,
        optimizer=tf.train.GradientDescentOptimizer)}, dsets)
