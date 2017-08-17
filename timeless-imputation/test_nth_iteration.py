import os
import sys
output = os.path.join(os.path.dirname(sys.argv[1]), "mf_out.pkl.gz")
assert not os.path.exists(output)
import pickle_utils as pu
import missForest

@pu.memoize("info_BostonHousing.pkl.gz")
def get_dataset_info(name="BostonHousing"):
    import category_dae, datasets
    ds = datasets.datasets()["BostonHousing"]
    info = category_dae.dataset_dimensions_info(ds)
    missForest.preprocess_dataframe(ds[0], info)
    return info

info = get_dataset_info()
df = missForest.postprocess_dataframe(pu.load(sys.argv[1])[0], info)
pu.dump([df], output)
