import os
import sys
import pickle_utils as pu
import utils
import missForest

path = sys.argv[1]
@pu.memoize("datasets.pkl.gz")
def get_datasets():
    import datasets
    return datasets.datasets()

ds = get_datasets()
fname = os.path.basename(path)
dirname = os.path.dirname(path)

for k, (d, _) in ds.items():
    idx = fname.find(k)
    if idx > 0:
        fname = "amputed_" + fname[idx:].rstrip("/") + ".pkl.gz"
        full_data = d
        data_name = k
        break

amputed_data = pu.load(os.path.join(dirname, fname))
_data, moments = utils.normalise_dataframes(amputed_data, full_data,
                                            method='mean_std')
_ad, _fd = _data
i = 0


@pu.memoize("info_{name:s}.pkl.gz")
def get_dataset_info(name="BostonHousing"):
    import category_dae, datasets
    ds = datasets.datasets()[name]
    info = category_dae.dataset_dimensions_info(ds)
    missForest.preprocess_dataframe(ds[0], info)
    return info

info = get_dataset_info(name=data_name)

while True:
    iter_fn = os.path.join(path, "iter_{:d}.pkl.gz".format(i))
    if not os.path.exists(iter_fn):
        break
    _id = missForest.postprocess_dataframe(pu.load(iter_fn)[0], info)
    imputed_data = utils.unnormalise_dataframes(moments, [_id])
    print("Iter", i, utils.reconstruction_metrics(amputed_data, full_data,
                                                  imputed_data))
    i += 1
