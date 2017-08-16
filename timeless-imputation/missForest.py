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
    def __init__(self, n_features, **kwargs):
        super(RF_class, self).__init__(
            n_estimators=100, n_jobs=-1,
            max_features=int(np.floor(n_features**.5)), bootstrap=False,
            min_samples_split=5)
    def predict_proba(self, X, X_var):
        p = super(RF_class, self).predict_proba(X)
        if p.shape[1] == 2:
            return p[:, 1:2], None
        return p, None

    def fit(self, X, X_var, y, optimize=None):
        return super(RF_class, self).fit(X, y)

    def predict(self, X, X_var):
        return super(RF_class, self).predict(X), None


class RF_reg(RandomForestRegressor):
    def __init__(self, n_features, **kwargs):
        super(RF_reg, self).__init__(
            n_estimators=100, n_jobs=-1,
            max_features=int(np.floor(n_features**.5)), bootstrap=False,
            min_samples_split=5)

    def predict(self, X, X_var):
        return super(RF_reg, self).predict(X), None

    def fit(self, X, X_var, y, optimize=None):
        return super(RF_reg, self).fit(X, y)


def predict(df, df_var, other_info, dense_df, prev_df, complete_df, train_mask,
            key, cat_dummies, classifier, regressor,
            use_previous_prediction=False, optimize=True, n_neighbours=None,
            knn_type=None, model_fname=None, gp_params=None, **kwargs):
    #if not use_previous_prediction:
    #    del prev_df  # To prevent bugs

    if key in cat_dummies:
        X = df.drop(cat_dummies[key], axis=1)
        #complete_X = complete_df.drop(cat_dummies[key], axis=1)
        X_var = df_var.drop(cat_dummies[key], axis=1)
        list_keys = list(df.keys())
        prev_i = None
        for c in cat_dummies[key]:
            i = list_keys.index(c)
            if prev_i is not None:
                assert i == prev_i + 1, "cat_dummy keys are contiguous"
            prev_i = i

        key_first = list_keys.index(cat_dummies[key][0])
        key_last = list_keys.index(cat_dummies[key][-1])

        #if use_previous_prediction:
            #X = pd.concat([X, prev_df[cat_dummies[key]]], axis=1)
        y = dense_df[key]
        method = classifier
    else:
        X = df.drop(key, axis=1)
        #complete_X = complete_df.drop(key, axis=1)
        if df_var is None:
            import pdb
            pdb.set_trace()
        X_var = df_var.drop(key, axis=1)
        key_first = key_last = list(df.keys()).index(key)
        y = df[key]
        method = regressor

    if other_info is not None:
        covs_columns = np.concatenate([
            other_info['covariances'][:, :key_first],
            other_info['covariances'][:, key_last+1:]], axis=1)
        covs = np.concatenate([
            covs_columns[:, :, :key_first],
            covs_columns[:, :, key_last+1:]], axis=2)

        mog = {'weights': other_info['weights'],
                'means': np.concatenate([
                    other_info['means'][:, :key_first],
                    other_info['means'][:, key_last+1:]], axis=1),
                'covariances': covs}
    else:
        mog = None

    rf = method(X.shape[1], mog=mog, complete_X=X, n_neighbours=n_neighbours,
                knn_type=knn_type, params=gp_params, **kwargs)
    if gp_params is not None:
        assert not optimize, "We're going to overwrite the parameters anyways"
    rf.fit(X[train_mask], X_var[train_mask], y[train_mask], optimize=optimize)
    if hasattr(rf, "m") and hasattr(rf.m.kern, "rbf"):
        pu.dump({"rbf_variance": np.array(rf.m.kern.rbf.variance),
                 "white_variance": np.array(rf.m.kern.white.variance),
                 "rbf_lengthscale": np.array(rf.m.kern.rbf.lengthscale)},
                model_fname)

    if key in cat_dummies:
        n_cats = y.max() + 1
        update_ks = cat_dummies[key]
        if len(cat_dummies[key]) == 1 and n_cats > 2:
            out = rf.predict(X, X_var)[:, np.newaxis]
        else:
            out = rf.predict_proba(X, X_var)
    else:
        update_ks = [key]
        out = rf.predict(X, X_var)
    if isinstance(out, tuple):
        to_return = []
        for i, o in enumerate(out):
            try:
                if o is not None and not np.any(np.isnan(o)):
                    to_return.append(pd.DataFrame(o, index=y.index,
                                                columns=update_ks))
            except Exception:
                import pdb
                pdb.set_trace()
        if len(to_return) > 1:
            return tuple(to_return)
        return to_return[0]
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
                 .clip(0, n_cats-1).astype(np.int32))
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

def mean_impute(log_path, df, full_data, info):
    """This automatically also puts the probabilities of unordered categoricals
    and it should be OK with ordered categoricals"""
    return (df.fillna(df.mean()),
            (df.applymap(lambda x: np.nan if np.isnan(x) else 0.0)
             .fillna(df.std())),
            None)
def no_impute(log_path, df, full_data, info):
    return (df,
            (df.applymap(lambda x: np.nan if np.isnan(x) else 0.0)
             .fillna(df.std())),
            None)

def KNN_GP_impute(log_path, df, info):
    name = os.path.basename(log_path.rstrip("/"))
    fname = name[name.find("BostonHousing"):]
    print(fname)
    long_path = pu.load("impute_benchmark_old/impute_benchmark/imputed_GP_KNN_kernel_5_iter00_{:s}/iter_1.pkl.gz".format(fname))
    return long_path + (None,)


def impute(log_path, dataset, full_data, sequential=True,
           predictors=(RF_class, RF_reg), initial_impute=mean_impute,
           ignore_ordered=True, print_progress=True, max_iterations=25,
           use_previous_prediction=False, impute_name_replace=None,
           optimize_gp=True, n_neighbours=5, knn_type='kernel_avg',
           load_gp_model=None, **kwargs):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    memoized_fname = os.path.join(log_path, "mf_out.pkl.gz")
    if os.path.exists(memoized_fname):
        data = pu.load(memoized_fname)
        if not isinstance(data, tuple):
            for i in range(50):
                f_path = os.path.join(log_path, "iter_{:d}.pkl.gz".format(i))
                if not os.path.exists(f_path):
                    i -= 1
                    f_path = os.path.join(log_path, "iter_{:d}.pkl.gz".format(i))
                    break
            assert i < 50, "impossible so many iterations"
            return data, pu.load(f_path)[1]
        return data

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
    test_df, test_df_var, test_df_other_first_iteration = initial_impute(
        impute_log_path, test_df, full_data[0], info)
    #test_df = full_data[0].copy()
    #test_df_var = test_df.applymap(lambda _: 0.0)
    #test_df_other_first_iteration = None

    # Sort by increasing amount of missingness
    #classifications = sorted(masks_usable.keys(),
    #                         key=lambda k: -masks_usable[k].sum())
    #masks_usable = masks_usable[classifications]

    if print_progress:
        print("Start", utils.reconstruction_metrics(
            dataset[0], full_data[0], postprocess_dataframe(test_df, info)))
    pu.dump((test_df, test_df_var), os.path.join(log_path, 'iter_0.pkl.gz'))

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
        for key in masks_usable.keys():
            mask = masks_usable[key]
            if load_gp_model is not None:
                fp = os.path.join(load_gp_model(log_path),
                                  ("model_{:s}_iter_{:d}.pkl.gz"
                                   .format(key, iter_i)))
                print("Loading parameters from", fp)
                gp_params = pu.load(fp)
            else:
                gp_params = None

            y = predict(test_df, test_df_var, test_df_other_first_iteration,
                        dense_df, predicted_df, full_data[0], mask, key,
                        info["cat_dummies"], classifier=predictors[0],
                        regressor=predictors[1],
                        use_previous_prediction=use_previous_prediction,
                        optimize=optimize_gp, n_neighbours=n_neighbours,
                        knn_type=knn_type,
                        model_fname=os.path.join(
                            log_path, 'model_{:s}_iter_{:d}.pkl.gz'.format(
                                key, iter_i)
                        ), gp_params=gp_params, **kwargs)
            print("model_RMSE:", utils.rmse([True]*np.sum(~mask),
                                         full_data[0].loc[~mask, key].values,
                                         [(y[0] if isinstance(y, tuple) else y)
                                          .values.flatten()[~mask]]))
            #rf_y = predict(test_df.fillna(0.0), test_df_var,
            #test_df_other_first_iteration,
            #               dense_df, predicted_df, full_data[0], mask, key,
            #               info["cat_dummies"], classifier=RF_class,
            #               regressor=RF_reg,
            #               use_previous_prediction=use_previous_prediction,
            #               optimize=optimize_gp)
            #print("rf_RMSE:", utils.rmse([True]*np.sum(~mask),
            #                             full_data[0].loc[~mask, key].values,
            #                             [(rf_y[0] if isinstance(rf_y, tuple) else rf_y)
            #                              .values.flatten()[~mask]]))
            if isinstance(y, tuple):
                y, y_var = y
            else:
                y_var = None

            if sequential:
                update_ks = list(y.keys())
                test_df.loc[~mask, update_ks] = y[~mask]
                if y_var is None:
                    test_df_var = None
                else:
                    test_df_var.loc[~mask, update_ks] = y_var[~mask]
                predicted_df.loc[:, update_ks] = y
                dense_df = postprocess_dataframe(test_df, info,
                                                 reindex_categories=False)
            else:
                updates.append((~mask, y, y_var))

        if not sequential:
            for _mask, y, y_var in updates:
                update_ks = list(y.keys())
                test_df.loc[_mask, update_ks] = y[_mask]
                if y_var is None:
                    test_df_var = None
                else:
                    test_df_var.loc[_mask, update_ks] = y_var[_mask]
                predicted_df.loc[:, update_ks] = y
            dense_df = postprocess_dataframe(test_df, info,
                                             reindex_categories=False)

        # The additional information, for example the MoG model, only lasts for
        # the first iteration.
        test_df_other_first_iteration = None

        sq_diff = (prev_nums - test_df[num_idx])**2
        cur_num_change = np.sum(sq_diff.values) / np.sum(
            test_df[num_idx].values**2)

        cur_cat_change = np.sum((dense_df[cat_idx] != prev_cats).values)

        if print_progress:
            print("Iter", iter_i, utils.reconstruction_metrics(
                dataset[0], full_data[0], postprocess_dataframe(test_df, info)),
                cur_num_change, cur_cat_change)
        pu.dump((test_df, test_df_var), os.path.join(
            log_path, 'iter_{:d}.pkl.gz'.format(iter_i)))

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
    return out, test_df_var


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
    dsets = dict((x, _ds[x]) for x in ["BostonHousing",
                                       #"Servo", # "BostonHousing", "Ionosphere",
                                       ])  # "BreastCancer", "Soybean"])


    #dsets = dict((x, _ds[x]) for x in ["Servo"])
    baseline = datasets.benchmark({
        #'MF_py': lambda log, d, full_data: impute(
        #    log, d, full_data, sequential=False, print_progress=True,
        #    use_previous_prediction=False),
        #'mean': lambda log, d, full_data: impute(
        #    log, d, full_data, max_iterations=0),
        #'GMM': lambda log, d, full_data: impute(
        #    log, d, full_data, max_iterations=0,
        #    initial_impute=mbm.mf_initial_impute),
        #'GMM_raw': lambda log, d, full_data: impute(
        #    log, d, full_data, max_iterations=0,
        #    impute_name_replace=('GMM_raw', 'GMM'),
        #    initial_impute=lambda *args, **kwargs: mbm.mf_initial_impute(
        #        *args, **kwargs, ignore_categories=True)),
        ##'MF_GMM': lambda log, d, full_data: impute(
        #    log, d, full_data, initial_impute=mbm.mf_initial_impute,
        #    sequential=False, print_progress=True,
        #    use_previous_prediction=False, impute_name_replace=('MF_GMM', 'GMM')),
        'MissForest_fulldata': datasets.memoize(utils.impute_missforest),
    }, dsets, do_not_compute=False)
    print(baseline)
