import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects
import rpy2.robjects.pandas2ri
import rpy2.robjects.numpy2ri
rpy2.robjects.pandas2ri.activate()
rpy2.robjects.numpy2ri.activate()

py2ri = rpy2.robjects.pandas2ri.py2ri
ri2py = rpy2.robjects.pandas2ri.ri2py

mice = importr('mice')
missForest = importr('missForest')
doParallel = importr('doParallel')
parallel = importr('parallel')
n_cores = parallel.detectCores()
doParallel.registerDoParallel(parallel.makeCluster(n_cores))

__all__ = ['rmse', 'mean_rmse', 'gondara_rmse_sum', 'mcar_missing_rows',
           'mcar_missing_total', 'normalise_dataframes', 'impute_mice',
           'impute_missforest']

def rmse(mask_missing, original_df, multiple_imputed_df):
    "Compute the RMSE between a dataset and its imputations"
    assert original_df.shape == multiple_imputed_df[0].shape, "data set shape not matching"
    sq_diff = (original_df-multiple_imputed_df)**2
    mse = np.sum(sq_diff*mask_missing)/np.sum(mask_missing)
    return mse**.5

def mean_rmse(mask_missing, original_df, multiple_imputed_df):
    "Compute RMSE of the optimal point according to the multiple-imputed distribution"
    midf = np.mean(multiple_imputed_df, axis=0, keepdims=True)
    return rmse(mask_missing, original_df, midf)

def gondara_rmse_sum(mask_missing, original_df, multiple_imputed_df):
    "RMSE_sum as in Multiple Imputation Using Deep Denoising Autoencoders (Gondara & Wang 2017)"
    assert original_df.shape == multiple_imputed_df[0].shape, "data set shape not matching"
    sq_diff = (original_df-multiple_imputed_df)**2
    sq_diff[:,~mask_missing] = 0
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
        d[i,r[:int(r.shape[0]*missing_row_loss)]] = np.nan
    return dataset

def mcar_rows(dataset_, rows_missing=0.2, missing_row_loss=0.5):
    dataset = dataset_.copy()
    missing = np.random.rand(len(dataset)) < rows_missing
    for i in np.nonzero(missing)[0]:
        row_missing = np.random.rand(dataset.shape[1]) < missing_row_loss
        if hasattr(dataset, 'values'):
            d = dataset.values
        else:
            d = dataset
        d[i,row_missing] = np.nan
    return dataset

def mcar_total(dataset_, missing_proportion=0.5):
    "Make each cell of the dataset missing uniformly at random"
    dataset = dataset_.copy()
    missing = np.random.rand(*dataset.shape) < missing_proportion
    if hasattr(dataset, 'values'):
        d = dataset.values
    else:
        d = dataset
    d[missing] = np.nan
    return dataset

def normalise_dataframes(*dataframes):
    "Normalise all passed dataframes with the mean and std of the first"
    mean = dataframes[0].mean()
    std = dataframes[0].std()
    return tuple(map(lambda d: (d-mean)/std, dataframes))

def impute_mice(dataset, number_imputations=5):
    "Imputed dataset using MICE"
    df = py2ri(dataset)
    obj = mice.mice(df, m=number_imputations, maxit=100, method='pmm', seed=500)
    return list(ri2py(mice.complete(obj, v)) for v in range(1, number_imputations+1))

def impute_missforest(dataset, number_imputations=1):
    assert number_imputations==1
    if dataset.shape[1] < n_cores[0]:
        par = 'no'
    else:
        par = 'variables'
    mf_imp = missForest.missForest(dataset, parallelize=par)
    return [ri2py(mf_imp[0])]
