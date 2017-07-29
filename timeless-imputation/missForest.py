import numpy as np
import datasets
import utils
import unittest
import category_dae
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


def predict_Py_RF(df, dense_df, prev_df, train_mask, key, cat_dummies,
                  use_previous_prediction=False):
    if not use_previous_prediction:
        del prev_df  # To prevent bugs

    if key in cat_dummies:
        X = df.drop(cat_dummies[key], axis=1)
        y = dense_df[key]
        rf = RandomForestClassifier()
    else:
        X = df.drop(key, axis=1)
        n_features = X.shape[1]  # TODO: not always true
        y = df[key]
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                   max_features=int(np.floor(n_features**.5)),
                                   bootstrap=False, min_samples_split=5)
    rf.fit(X[train_mask], y[train_mask])

    if key in cat_dummies:
        n_cats = y.max() + 1
        update_ks = cat_dummies[key]
        if len(cat_dummies[key]) == 1:
            if n_cats > 2:
                out = rf.predict(X)[:, np.newaxis]
            else:
                out = rf.predict_proba(X)[:, 1:2]
        else:
            out = rf.predict_proba(X)
    else:
        update_ks = [key]
        out = rf.predict(X)
    return pd.DataFrame(out, index=y.index, columns=update_ks)


def predict_R_RF(df, dense_df, prev_df, train_mask, key, cat_dummies):
    X = dense_df.drop(key, axis=1)
    y = utils.R_random_forest(X[train_mask].values,
                              dense_df.loc[train_mask, key].values,
                              X.values)
    return pd.DataFrame(y, index=X.index, columns=[key])


def preprocess_dataframe(df, info, ignore_ordered=False,
                         reindex_categories=True):
    if reindex_categories:
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


def postprocess_dataframe(df, info, reindex_categories=True):
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
    if reindex_categories:
        out = category_dae.postprocess_dataframe(out, info)
    for k in nans.keys():
        out.loc[nans[k], k] = category_dae.NA_int32
    return out


def impute(dataset, full_data, sequential=True, predict_fun=predict_Py_RF,
           ignore_ordered=True, max_iterations=25):
    info = category_dae.dataset_dimensions_info(dataset)
    test_df, masks_usable = preprocess_dataframe(
        dataset[0], info, ignore_ordered=ignore_ordered)

    # Tenatively fill NAs
    # This automatically also puts the probabilities of unordered categoricals
    # and it should be OK with ordered categoricals
    test_df = test_df.fillna(test_df.mean())

    # Sort by increasing amount of missingness
    classifications = sorted(masks_usable.keys(),
                             key=lambda k: -masks_usable[k].sum())
    masks_usable = masks_usable[classifications]

    print("Start", utils.reconstruction_metrics(
        dataset[0], full_data[0], postprocess_dataframe(test_df, info)))

    num_idx = info["num_idx"]
    cat_idx = info["cat_idx"]

    # Now perform the imputation procedure
    prev_num_change = prev_cat_change = np.inf
    predicted_df = test_df.copy()
    dense_df = postprocess_dataframe(test_df, info, reindex_categories=False)
    prev_nums = dense_df[num_idx].copy()
    prev_cats = dense_df[cat_idx].copy()
    for iter_i in range(1, max_iterations+1):
        updates = []
        for key in classifications:
            mask = masks_usable[key]
            y = predict_fun(test_df, dense_df, predicted_df, mask, key,
                            info["cat_dummies"])

            if sequential:
                update_ks = list(y.keys())
                test_df.loc[~mask, update_ks] = y[~mask]
                predicted_df.loc[:, update_ks] = y
                dense_df = postprocess_dataframe(test_df, info,
                                                 reindex_categories=False)
            else:
                updates.append((~mask, y))

        if not sequential:
            for _mask, y in updates:
                update_ks = list(y.keys())
                test_df.loc[_mask, update_ks] = y[_mask]
                predicted_df.loc[:, update_ks] = y
            dense_df = postprocess_dataframe(test_df, info,
                                             reindex_categories=False)

        sq_diff = (prev_nums - test_df[num_idx])**2
        cur_num_change = np.sum(sq_diff.values) / np.sum(
            test_df[num_idx].values**2)

        cur_cat_change = np.sum((dense_df[cat_idx] != prev_cats).values)

        print("Iter", iter_i, utils.reconstruction_metrics(
            dataset[0], full_data[0], postprocess_dataframe(test_df, info)),
              cur_num_change, cur_cat_change)

        if (cur_num_change < prev_num_change or
                cur_cat_change < prev_cat_change):
            prev_num_change = cur_num_change
            prev_cat_change = cur_cat_change
            prev_nums = dense_df[num_idx].copy()
            prev_cats = dense_df[cat_idx].copy()
        else:
            dense_df = pd.concat([prev_nums, prev_cats], axis=1)[
                list(dense_df.keys())]
            break
    return [postprocess_dataframe(preprocess_dataframe(
        dense_df, info, ignore_ordered=ignore_ordered,
        reindex_categories=False)[0], info)]


class TestPrePostprocessing(unittest.TestCase):
    def test_values_reconstructed(self):
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
    dsets = dict((x, _ds[x]) for x in ["Soybean"])
    baseline = datasets.benchmark({
        'MF_Py_par': lambda _path, d, full_data: impute(
            d, full_data, sequential=False),
        'MF_Py_seq': lambda _path, d, full_data: impute(
            d, full_data, sequential=True),
        'MissForest': datasets.memoize(utils.impute_missforest)
    }, dsets, do_not_compute=False)
    print(baseline)
