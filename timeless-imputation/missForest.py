import numpy as np
import datasets
import utils
import unittest
import category_dae
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import os
import pickle_utils as pu


class RF_class(RandomForestClassifier):
    def __init__(self, n_features):
        super(RF_class, self).__init__(
            n_estimators=100, n_jobs=-1,
            max_features=int(np.floor(n_features**.5)), bootstrap=False,
            min_samples_split=5)
    def predict_proba(self, X):
        p = super(RF_class, self).predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1:2]
        return p


def RF_reg(n_features):
    return RandomForestRegressor(
        n_estimators=100, n_jobs=-1,
        max_features=int(np.floor(n_features**.5)), bootstrap=False,
        min_samples_split=5)


def predict(df, dense_df, prev_df, train_mask, key, cat_dummies,
            classifier, regressor, use_previous_prediction=False, df_var=None):
    if not use_previous_prediction:
        del prev_df  # To prevent bugs

    if key in cat_dummies:
        X = df.drop(cat_dummies[key], axis=1)
        if use_previous_prediction:
            X = pd.concat([X, prev_df[cat_dummies[key]]], axis=1)
        y = dense_df[key]
        rf = classifier(X.shape[1])
    else:
        X = df.drop(key, axis=1)
        if use_previous_prediction:
            X = pd.concat([X, prev_df[[key]]], axis=1)
        y = df[key]
        rf = regressor(X.shape[1])
    rf.fit(X[train_mask], y[train_mask])

    if key in cat_dummies:
        n_cats = y.max() + 1
        update_ks = cat_dummies[key]
        if len(cat_dummies[key]) == 1 and n_cats > 2:
            out = rf.predict(X)[:, np.newaxis]
        else:
            out = rf.predict_proba(X)
    else:
        update_ks = [key]
        out = rf.predict(X)
    return pd.DataFrame(out, index=y.index, columns=update_ks)


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
        if (n_dims == 1 and not ignore_ordered) or n_cats == 2:
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
        n_cats = info["n_cats_l"][info["cat_idx"].index(k_orig)]
        nans[k_orig] = df[k_dum[0]].isnull()
        if (n_dims == 1 and not info["ignore_ordered"]) or n_cats == 2:
            assert len(k_dum) == 1
            f = (df[k_dum[0]].where(np.isfinite, other=0).round()
                 .astype(np.int32))
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

def mean_impute(log_path, df, info):
    """This automatically also puts the probabilities of unordered categoricals
    and it should be OK with ordered categoricals"""
    return df.fillna(df.mean())


def impute(log_path, dataset, full_data, sequential=True,
           predictors=(RF_class, RF_reg), initial_impute=mean_impute,
           ignore_ordered=True, print_progress=True, max_iterations=25,
           use_previous_prediction=False, impute_name_replace=None):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    memoized_fname = os.path.join(log_path, "mf_out.pkl.gz")
    if os.path.exists(memoized_fname):
        return pu.load(memoized_fname)

    info = category_dae.dataset_dimensions_info(dataset)
    test_df, masks_usable = preprocess_dataframe(
        dataset[0], info, ignore_ordered=ignore_ordered)

    # Tenatively fill NAs
    if impute_name_replace is None:
        impute_log_path = log_path
    else:
        impute_log_path = log_path.replace(*impute_name_replace)
    if not os.path.exists(impute_log_path):
        os.mkdir(impute_log_path)
    test_df = initial_impute(impute_log_path, test_df, info)

    # Sort by increasing amount of missingness
    classifications = sorted(masks_usable.keys(),
                             key=lambda k: -masks_usable[k].sum())
    masks_usable = masks_usable[classifications]

    if print_progress:
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
            y = predict(test_df, dense_df, predicted_df, mask, key,
                        info["cat_dummies"], classifier=RF_class,
                        regressor=RF_reg,
                        use_previous_prediction=use_previous_prediction)
            #print("RF_RMSE:", utils.rmse([True]*np.sum(~mask),
            #                             full_data[0].loc[~mask, key].values,
            #                             [y[~mask].values.flatten()]))

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

        if print_progress:
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
    out = [postprocess_dataframe(preprocess_dataframe(
        dense_df, info, ignore_ordered=ignore_ordered,
        reindex_categories=False)[0], info)]
    pu.dump(out, memoized_fname)
    return out


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
    import missing_bayesian_mixture as mbm

    _ds = datasets.datasets()
    #dsets = dict((x, _ds[x]) for x in ["Soybean", "BostonHousing",
    #                                   "BreastCancer", "Ionosphere", "Servo"])
    dsets = dict((x, _ds[x]) for x in ["BostonHousing", "Ionosphere"])
    baseline = datasets.benchmark({
        #'MF_Py_par': datasets.memoize(lambda d, full_data: impute(
        #    d, full_data, sequential=False, print_progress=True)),
        #'MF_Py_seq': datasets.memoize(lambda d, full_data: impute(
        #    d, full_data, sequential=True, print_progress=True)),
        'MF_py': lambda log, d, full_data: impute(
            log, d, full_data, sequential=False, print_progress=True,
            use_previous_prediction=False),
        'mean': lambda log, d, full_data: impute(
            log, d, full_data, max_iterations=0),
        'GMM': lambda log, d, full_data: impute(
            log, d, full_data, max_iterations=0,
            initial_impute=mbm.mf_initial_impute),
        #'GMM_5res': lambda log, d, full_data: impute(
        #    log, d, full_data, max_iterations=0,
        #    initial_impute=mbm.mf_initial_impute),
        #'MF_GMM_5res': lambda log, d, full_data: impute(
        #    log, d, full_data, initial_impute=mbm.mf_initial_impute,
        #    sequential=False, print_progress=True,
        #    use_previous_prediction=False, impute_name_replace=('MF_GMM_5res', 'GMM_5res')),
        'MF_GMM': lambda log, d, full_data: impute(
            log, d, full_data, initial_impute=mbm.mf_initial_impute,
            sequential=False, print_progress=True,
            use_previous_prediction=False, impute_name_replace=('MF_GMM', 'GMM')),
        #'BGMM_20': lambda p, d, full_data: mbm.impute_bayes_gmm(
        #    p, d, full_data=full_data, number_imputations=100,
        #    n_components=20),
        'MissForest': datasets.memoize(utils.impute_missforest),
        #'GP_naive': datasets.memoize(lambda d, full_data: impute(
        #    d, full_data, sequential=False, print_progress=True,
        #    predictors=(mfGP.classifier, mfGP.regressor))),
        #'MissForest': datasets.memoize(utils.impute_missforest)
    }, dsets, do_not_compute=False)
    print(baseline)
