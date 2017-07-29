import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
import collections

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

__all__ = ['rmse', 'mean_rmse', 'gondara_rmse_sum', 'mcar_rows', 'mcar_total',
           'normalise_dataframes', 'impute_mice', 'impute_missforest',
           'df_from_R', 'df_to_R', 'NA_int32']

NA_int32 = -(1 << 31)


def df_to_R(dataset, name):
    df, category_keys = dataset
    int_keys = list(filter(lambda k: (df[k].dtype == np.int32 and
                                      k not in category_keys), df.keys()))
    R.assign(name, df)
    keys = int_keys + category_keys
    for i in range(0, len(keys), 10):
        R("{:s} <- transform({:s}, {:s})".format(
            name, name, ", ".join(
                "{0:s}=as.ordered({1:s}${0:s})".format(k, name)
                for k in keys[i:i + 10])))
    for k in category_keys:
        R('class({:s}${:s}) <- "factor"'.format(name, k))


def df_from_R(name, more_than_one_value=False):
    # First pass: get which keys are objects
    df = R(name)
    category_keys = list(filter(lambda k: df[k].dtype == np.object, df.keys()))

    # Second pass: get these keys as ordered, to get the NAs correctly
    if len(category_keys) > 0:
        df = R("transform({:s}, {:s})".format(name, ", ".join(
            "{:s}=as.ordered({:s}${:s})".format(k, name, k)
            for k in category_keys)))
    else:
        df = R(name)
    assert all(df[k].dtype in [np.int32, np.float64] for k in df.keys())

    for k in filter(lambda k: df[k].dtype == np.int32, df.keys()):
        unique = list(df[k].unique())
        if NA_int32 in unique:
            unique.remove(NA_int32)
        unique.sort()

        msg = ""
        if unique[0] != 1:
            msg += "categories start counting at 1"
        if more_than_one_value and len(unique) <= 1:
            msg += "; must have more than 1 possible value"
        for i in range(1, len(unique)):
            if unique[i] != unique[i-1]+1:
                msg += "; must be counting up contiguously"
                break
        if msg != "":
            print(k, msg)
            df = df.drop(k, axis=1)
            category_keys.remove(k)

    return df, category_keys


def rmse(mask_missing, original_df, multiple_imputed_df):
    "Compute the RMSE between a dataset and its imputations"
    assert original_df.shape == multiple_imputed_df[0].shape, \
        "data set shape not matching"
    total = np.sum(mask_missing)
    if total == 0:
        return np.nan
    sq_diff = (original_df-multiple_imputed_df)**2
    mse = np.sum(sq_diff*mask_missing) / total
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
    m_df = pd.DataFrame(~missing_mask, columns=list(dataset.keys()))
    df = pd.concat([
        dataset[int_keys].where(m_df[int_keys].values, other=NA_int32),
        dataset[float_keys].where(m_df[float_keys].values),
        ], axis=1)
    return df[dataset.keys()]


def mcar_rows(dataset, rows_missing=0.2, missing_row_loss=0.5):
    """Make each row missing with probability `rows_missing`. Rows that are
    missing have a `missing_row_loss` probability of having each of their
    elements missing. Thus the proportion of missing values in total is
    `rows_missing * missing_row_loss`"""
    example_mask = np.random.rand(dataset.shape[0]) < rows_missing
    row_mask = np.random.rand(*dataset.shape) < missing_row_loss
    overall_mask = example_mask[:, np.newaxis] & row_mask
    return _type_aware_drop(dataset, overall_mask)


def mcar_total(dataset, missing_proportion=0.5):
    "Make each cell of the dataset missing uniformly at random"
    return _type_aware_drop(dataset, (np.random.rand(*dataset.shape)
                                      < missing_proportion))


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


def impute_mice(dataset, number_imputations=5, method='pmm', full_data=None):
    "Imputed dataset using MICE"
    del full_data
    df_to_R(dataset, "df")
    R("imputed_df <- mice(df, m={:d}, maxit=100, method='{:s}', seed={:d})"
      .format(number_imputations, method, 500))
    dfs = []
    for i in range(1, number_imputations + 1):
        R("idf <- complete(imputed_df, {:d})".format(i))
        dfs.append(df_from_R("idf")[0])
    return dfs


def impute_missforest(dataset, number_imputations=1, full_data=None):
    del full_data
    assert number_imputations == 1
    if dataset[0].shape[1] < n_cores[0]:
        par = 'no'
    else:
        par = 'variables'
    df_to_R(dataset, "df")
    R("imputed_df <- missForest(df, parallelize='{:s}')".format(par))
    df, cat_idx = df_from_R("imputed_df$ximp")
    return [df]


def R_random_forest(x, y, misX, ntree=100):
    R.assign("x", x)
    R.assign("y", y)
    sampsize = x.shape[0]
    nodesize = 5
    mtry = int(np.floor(x.shape[1]**.5))
    R("""RF <- randomForest(x = x, y = y, ntree = {ntree:d}, mtry = {mtry:d},
          replace = TRUE, sampsize = {sampsize:d},
          nodesize = {nodesize:d}, maxnodes=NULL)""".format(
              ntree=ntree, mtry=mtry, sampsize=sampsize,
              nodesize=nodesize))
    R.assign("misX", misX)
    R("misY <- predict(RF, misX)")
    return R("misY")


def dataframe_like(dataframe, new_values):
    return pd.DataFrame(new_values, index=dataframe.index,
                        columns=dataframe.columns)


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
    int_mask = (amputed_data[int_keys].values == NA_int32)
    int_mask &= (full_data[int_keys].values != NA_int32)
    wrong, total = count_wrong_classifications(
        full_data[int_keys].values[int_mask],
        list(i[int_keys].values[int_mask] for i in imputed_data))
    if total == 0:
        return np.nan
    return wrong/total


def normalised_rmse(mask_missing, original_df, multiple_imputed_df=None,
                    rmse_val=None):
    """Normalised RMSE:
    For normalised RMSE, we take mean_std normalisation over the missing
    values only"""
    if rmse_val is None:
        rmse_val = rmse(mask_missing, original_df, multiple_imputed_df)
    arr = original_df[mask_missing]
    if len(arr) == 0:
        return np.nan
    return rmse_val / np.std(arr)


def reconstruction_metrics(amputed_data, full_data, imputed_data):
    # RMSE of 0-1 normalised data
    if not isinstance(imputed_data, list):
        imputed_data = [imputed_data]
    dfs, moments = normalise_dataframes(full_data, amputed_data, *imputed_data,
                                        method='min_max')
    rmse_fd, rmse_ad, *rmse_id = dfs
    rmse_ad, rmse_fd, rmse_id = amputed_data, full_data, imputed_data

    numerical_keys = list(moments[0].keys())
    multiple_imputed_array = list(d[numerical_keys].values for d in rmse_id)
    imputed_array = np.mean(multiple_imputed_array, axis=0, keepdims=True)
    rmse_args = (np.isnan(rmse_ad[numerical_keys].values),
                 rmse_fd[numerical_keys].values,
                 imputed_array)

    d = {'RMSE': mean_rmse(*rmse_args)}
    d['NRMSE'] = normalised_rmse(*rmse_args, rmse_val=d['RMSE'])
    d['PFC'] = percentage_falsely_classified(amputed_data, full_data,
                                             imputed_data)
    return d
