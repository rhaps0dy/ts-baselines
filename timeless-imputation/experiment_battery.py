import datasets
import utils
import missing_bayesian_mixture as mbm
import missForest
import numpy as np
import missForest_GP

_ds = datasets.datasets()

# dsets = dict((x, (_ds[x][0].drop(list(filter(
#     lambda k: _ds[x][0][k].dtype == np.int32, _ds[x][0].keys())), axis=1),
#     [])) for x in ["BostonHousing"])

dsets = dict((x, _ds[x]) for x in ["BostonHousing", "Ionosphere"])
# "Servo", "Soybean", "BreastCancer",

tests_to_perform = []
for b in datasets.items():
    def do_mcar_rows(dataset_, proportion):
        return utils.mcar_rows(dataset_, proportion**.5, proportion**.5)
    for c in [(datasets.memoize(utils.mcar_total), 'MCAR_total'),
              (datasets.memoize(do_mcar_rows), 'MCAR_rows')]:
        for d in [.1, .3, .5, .7, .9]:
            for e in ['mean_std']:
                tests_to_perform.append((b, c, d, e))
del b, c, d, e

#np.random.shuffle(tests_to_perform)

baseline = datasets.benchmark({
    'MissForest': datasets.memoize(utils.impute_missforest),
    'MICE': datasets.memoize(utils.impute_mice),
    'GMM': lambda log, d, full_data: missForest.impute(
        log, d, full_data, max_iterations=0,
        initial_impute=mbm.mf_initial_impute),

    'GP_mog_unopt': lambda log_path, d, full_data: missForest.impute(
        log_path, d, full_data, sequential=False, print_progress=True,
        predictors=(missForest_GP.UncertainGPClassification,
                    missForest_GP.UncertainGPRegression),
        optimize_gp=False, use_previous_prediction=False,
        ARD=False, impute_name_replace=('GP', 'GMM'), max_iterations=1),

    'GP_KNN': lambda log_path, d, full_data: missForest.impute(
        log_path, d, full_data, sequential=False, print_progress=True,
        predictors=(missForest_GP.KNNGPClassification,
                    missForest_GP.KNNGPRegression),
        optimize_gp=True, use_previous_prediction=False,
        ARD=True, n_neighbours=5, knn_type='kernel_avg', max_iterations=1),

    'mean': lambda log, d, full_data: missForest.impute(
        log, d, full_data, max_iterations=0),

    }, dsets, do_not_compute=False)

print(baseline)
