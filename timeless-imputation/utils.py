import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri

rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()
py2ri = rpy2.robjects.pandas2ri.py2ri
ri2py = rpy2.robjects.pandas2ri.ri2py
R = rpy2.robjects.r

base = importr('base')
base = importr('base')
mice = importr('mice')
missForest = importr('missForest')
doParallel = importr('doParallel')
parallel = importr('parallel')

n_cores = parallel.detectCores()
doParallel.registerDoParallel(parallel.makeCluster(n_cores))

__all__ = ['rmse', 'mean_rmse', 'gondara_rmse_sum', 'mcar_missing_rows',
           'mcar_missing_total', 'normalise_dataframes', 'impute_mice',
           'impute_missforest', 'df_py2ri', 'NA_INT32']

NA_INT32 = -(1 << 31)


def df_py2ri(df):
    ordered_keys = filter(lambda k: df[k].dtype == np.int32, df.keys())
    return base.transform(df, **dict((k, base.as_ordered(df[k]))
                                     for k in ordered_keys))


def df_ri2py(rdf):
    df = ri2py(rdf)
    for k in df.keys():
        assert df[k].dtype not in [np.float32, np.int64], \
            "f32 and i64 not handled"
        if df[k].dtype == np.object:
            df[k] = df[k].astype('category')
        if df[k].dtype == np.int32:
            assert not np.any(df[k].values == 0), \
                "Ordered variables should start at 1"
            df[k].values[df[k].values < 0] = 0
    return df


def rmse(mask_missing, original_df, multiple_imputed_df):
    "Compute the RMSE between a dataset and its imputations"
    assert original_df.shape == multiple_imputed_df[0].shape, \
        "data set shape not matching"
    sq_diff = (original_df-multiple_imputed_df)**2
    mse = np.sum(sq_diff*mask_missing)/np.sum(mask_missing)
    return mse**.5


def mean_rmse(mask_missing, original_df, multiple_imputed_df):
    """Compute RMSE of the optimal point according to the multiple-imputed
    distribution"""
    midf = np.mean(multiple_imputed_df, axis=0, keepdims=True)
    return rmse(mask_missing, original_df, midf)


def gondara_rmse_sum(mask_missing, original_df, multiple_imputed_df):
    """RMSE_sum as in Multiple Imputation Using Deep Denoising Autoencoders
    (Gondara & Wang 2017)"""
    assert original_df.shape == multiple_imputed_df[0].shape, \
        "data set shape not matching"
    sq_diff = (original_df-multiple_imputed_df)**2
    sq_diff[:, ~mask_missing] = 0
    per_attribute_rmse = np.mean(np.sum(sq_diff, axis=1), axis=0)**.5
    assert per_attribute_rmse.shape == (original_df.shape[1],)
    return np.sum(per_attribute_rmse)


def mcar_rows_gondara(dataset_, rows_missing=0.2, missing_row_loss=0.5):
    dataset = dataset_.copy()
    missing = np.random.rand(len(dataset)) < rows_missing
    for i in np.nonzero(missing)[0]:
        r = np.arange(dataset.shape[1])
        np.random.shuffle(r)
        print("dataset type", type(dataset))
        if hasattr(dataset, 'values'):
            d = dataset.values
        else:
            d = dataset
        d[i, r[:int(r.shape[0]*missing_row_loss)]] = np.nan
    return dataset


def _type_aware_drop(dataset, missing_mask):
    assert isinstance(dataset, pd.DataFrame)
    int_keys = list(filter(lambda k: dataset[k].dtype == np.int32,
                           dataset.keys()))
    float_keys = list(filter(lambda k: dataset[k].dtype == np.float64,
                             dataset.keys()))
    cat_keys = list(filter(lambda k: dataset[k].dtype.name == 'category',
                           dataset.keys()))
    m_df = pd.DataFrame(missing_mask, columns=list(dataset.keys()))
    df_cats = dataset[cat_keys].copy()
    for idx in zip(*np.nonzero(m_df[cat_keys].values)):
        df_cats.iloc[idx] = None
    df = pd.concat([
        dataset[int_keys].where(m_df[int_keys].values, other=NA_INT32),
        dataset[float_keys].where(m_df[float_keys].values),
        df_cats,
        ], axis=1)
    return df[dataset.keys()]


def mcar_rows(dataset, rows_missing=0.2, missing_row_loss=0.5):
    """Make each row missing with probability `rows_missing`. Rows that are
    missing have a `missing_row_loss` probability of having each of their
    elements missing. Thus the proportion of missing values in total is
    `rows_missing * missing_row_loss`"""
    example_mask = np.random.rand(dataset.shape[0]) < rows_missing
    row_mask = np.random.rand(*dataset.shape) < missing_row_loss
    overall_mask = example_mask[:, np.newaxis]
    return _type_aware_drop(dataset, row_mask*overall_mask)


def mcar_total(dataset, missing_proportion=0.5):
    "Make each cell of the dataset missing uniformly at random"
    return _type_aware_drop(dataset,
                            np.random.rand(*dataset.shape) < missing_proportion)


def _rescale_dataframes(dataframes, mean, std, rescale_f):
    keys = list(mean.keys())
    rescaled_dfs = list(pd.concat([
        rescale_f(df[keys], mean, std),
        df.drop(keys, axis=1)], axis=1) for df in dataframes)
    return rescaled_dfs


def normalise_dataframes(*dataframes, method='mean_std'):
    """Normalise all passed dataframes with the mean and std of the first, or
    by the min and max of the first."""
    df = dataframes[0]
    assert all(sorted(df.keys()) == sorted(a.keys())
               for a in dataframes), "All dataframes must have the same columns"
    numerical_columns = list(filter(lambda k: df[k].dtype == np.float64,
                                    df.keys()))
    if method == 'mean_std':
        mean = df[numerical_columns].mean(axis=0)
        std = df[numerical_columns].std(axis=0)
    elif method == 'min_max':
        mean = df[numerical_columns].min(axis=0)
        std = df[numerical_columns].max(axis=0) - mean
    else:
        raise ValueError("`method` must be one of ('min_max', 'mean_std')")
    return (_rescale_dataframes(dataframes, mean, std, lambda d, m, s: (d-m)/s),
            (mean, std))


def unnormalise_dataframes(mean_std, dataframes):
    """Return the dataframes to their original size using the information in
    `mean_std`"""
    mean, std = mean_std
    assert all(isinstance(df, pd.DataFrame) for df in dataframes)
    return _rescale_dataframes(dataframes, mean, std, lambda d, m, s: d*s+m)


def impute_mice(dataset, number_imputations=5, full_data=None):
    "Imputed dataset using MICE"
    del full_data
    df = df_py2ri(dataset)
    obj = mice.mice(df, m=number_imputations, maxit=100, method='pmm', seed=500)
    return list(ri2py(mice.complete(obj, v))
                for v in range(1, number_imputations + 1))


def impute_missforest(dataset, number_imputations=1, full_data=None):
    del full_data
    assert number_imputations == 1
    if dataset.shape[1] < n_cores[0]:
        par = 'no'
    else:
        par = 'variables'
    df = df_py2ri(dataset)
    mf_imp = missForest.missForest(df, parallelize=par)
    return [ri2py(mf_imp[0])]
