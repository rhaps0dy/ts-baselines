import numpy as np
import pickle_utils as pu
import datasets
import utils
import unittest
import category_dae
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def rf_impute(__path, dataset, full_data, *__args, **__kwargs):
    print("RFIMPUTE")
    @pu.memoize("mf.pkl.gz")
    def f():
        return utils.impute_missforest(dataset)
    a = f()
    print("RMSE:", utils.rmse(
        dataset[0].isnull().values, full_data[0].values, a))
    return 0


def predict_Py_RF(data, train_mask, j):
    X_ = np.concatenate([data.values[:, :j], data.values[:, j+1:]],
                        axis=1)
    X_train = X_[train_mask, :]
    y_train = data.values[train_mask, j]
    X_test = X_[~train_mask, :]
    y_test = data.values[~train_mask, j]
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,
                               max_features=int(np.floor(X_train.shape[1]**.5)),
                               bootstrap=False,
                               min_samples_split=5)
    rf.fit(X_train, y_train)
    train_perf = rf.score(X_train, y_train)
    test_perf = rf.score(X_test, y_test)
    print("Test: {:.4f}, Training: {:.4f}".format(test_perf,
                                                  train_perf))
    # Now fill the matrix with the predicted values
    y = rf.predict(X_test)
    return y


def predict_R_RF(data, train_mask, j):
    k = data.keys()[j]
    y = utils.R_random_forest(data.drop(k, axis=1).values[train_mask],
                              data[k].values[train_mask],
                              data.drop(k, axis=1).values[~train_mask])
    return y


def preprocess_dataframe(df, info, ignore_ordered=False):
    df = category_dae.preprocess_dataframe(df.copy(), info)

    masks_usable = pd.DataFrame()
    for k in info["num_idx"]:
        masks_usable[k] = ~df[k].isnull()

    info["ignore_ordered"] = ignore_ordered
    info["cat_dummies"] = {}
    final_df = [df[info["num_idx"]]]
    for k, n_dims, n_cats in zip(info["cat_idx"], info["n_dims_l"],
                                 info["n_cats_l"]):
        def to_float(n):
            if n == category_dae.post_NA_int32:
                return np.nan
            return float(n)
        masks_usable[k] = df[k] != category_dae.post_NA_int32

        s = df[k].map(to_float)
        if n_dims == 1 and not ignore_ordered:
            s = pd.DataFrame(s)
        else:
            s = pd.get_dummies(s, prefix=k).astype(np.float64)
        s.loc[~masks_usable[k], :] = np.nan
        info["cat_dummies"][k] = list(s.keys())
        final_df.append(s)

    return pd.concat(final_df, axis=1), masks_usable


def postprocess_dataframe(df, info):
    out = df[info["num_idx"]]
    nans = pd.DataFrame()
    for k_orig, k_dum in info["cat_dummies"].items():
        n_dims = info["n_dims_l"][info["cat_idx"].index(k_orig)]
        nans[k_orig] = df[k_dum[0]].isnull()
        if n_dims == 1 and not info["ignore_ordered"]:
            assert len(k_dum) == 1
            f = df[k_dum[0]].where(np.isfinite, other=0).astype(np.int32)
        else:
            k_dum_d = dict(zip(k_dum, range(len(k_dum))))
            # Set to 0 temporarily to be a valid category,
            # it will be set to post_NA_int32 later
            k_dum_d[np.nan] = 0
            f = (df[k_dum].apply(np.argmax, axis=1).map(k_dum_d)
                 .astype(np.int32))
        out.loc[:, k_orig] = f
    out = category_dae.postprocess_dataframe(out, info)
    for k in nans.keys():
        out.loc[nans[k], k] = category_dae.NA_int32
    return out


def impute(dataset, full_data, sequential=True):
    info = category_dae.dataset_dimensions_info(dataset)
    test_df, masks_usable = preprocess_dataframe(dataset[0], info)

    # Tenatively fill NAs
    test_df = test_df.fillna(test_df.mean())

    # Sort by increasing amount of missingness
    classifications = sorted(test_df.keys(),
                             key=lambda k: -masks_usable[k].sum())
    test_df = test_df[classifications]
    masks_usable = masks_usable[classifications]
    full_data = (full_data[0][classifications], full_data[1])

    # Now perform the imputation procedure
    print("RMSE {:d}:".format(0), utils.rmse(
        ~(masks_usable.values), full_data[0].values, [test_df.values]))
    prev_change = np.inf
    for iter_i in range(1, 50):
        updates = []
        prev_vals = test_df.values.copy()
        for j, key in enumerate(classifications):
            mask = masks_usable[key].values
            assert np.all(mask == masks_usable.values[:, j])
            y = predict_Py_RF(test_df, mask, j)
            if sequential:
                test_df.values[~masks_usable.values[:, j], j] = y
            else:
                updates.append(y)
            # If one by one:
            # test_imp_num[~mask, j] = y
            # print("\t\trmse:", utils.rmse(
            #     ~(masks_usable.values), full_data[0].values, [test_imp_num]))

        # updates is empty if not sequential
        for j, y in enumerate(updates):
            test_df.values[~masks_usable.values[:, j], j] = y

        cur_change = np.sum((prev_vals-test_df.values)**2) / np.sum(
            test_df.values**2)
        if cur_change > prev_change:
            test_df.values[...] = prev_vals
            break
        else:
            prev_change = cur_change
        print("RMSE {:d}:".format(iter_i), utils.rmse(
            ~(masks_usable.values), full_data[0].values, [test_df.values]))
    print("RMSE Final:".format(iter_i), utils.rmse(
        ~(masks_usable.values), full_data[0].values, [test_df.values]))
    # TODO: genralise
    test_df.chas = test_df.chas.map(np.floor).astype(np.int32)
    return list(map(
            lambda df: postprocess_dataframe(df, info),
            [test_df]))


class TestPrePostprocessing(unittest.TestCase):
    def test_values(self):
        dsets = datasets.datasets()
        for _, d in dsets.items():
            info = category_dae.dataset_dimensions_info(d)
            df, _ = preprocess_dataframe(d[0], info)
            pp = postprocess_dataframe(df, info)
            self.assertEqual(sorted(pp.keys()), sorted(d[0].keys()))
            for k in pp.keys():
                self.assertEqual(pp[k].dtype, d[0][k].dtype)
                self.assertTrue(np.all(pp[k].values == d[0][k].values))


if __name__ == '__main__':
    _ds = datasets.datasets()
    dsets = dict((x, _ds[x]) for x in ["BostonHousing"])
    baseline = datasets.benchmark({
        #'MF_R_seq': datasets.memoize(lambda d, full_data: impute(
        #    d, full_data, sequential=True)),
        'MF_R_par': datasets.memoize(lambda d, full_data: impute(
            d, full_data, sequential=False)),
    }, dsets, do_not_compute=False)
