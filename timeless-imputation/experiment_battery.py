import datasets
import utils
import missing_bayesian_mixture as mbm
import missForest
import numpy as np
import missForest_GP
import os

_ds = datasets.datasets()

# dsets = dict((x, (_ds[x][0].drop(list(filter(
#     lambda k: _ds[x][0][k].dtype == np.int32, _ds[x][0].keys())), axis=1),
#     [])) for x in ["BostonHousing"])

dsets = list((x, _ds[x]) for x in ["BostonHousing", "Ionosphere"])
# "Servo", "Soybean", "BreastCancer"

tests_to_perform = []
for b in dsets:
    def do_mcar_rows(dataset_, proportion):
        return utils.mcar_rows(dataset_, proportion**.5, proportion**.5)
    def do_mar_rows(dataset_, proportion):
        return utils.mar_rows(dataset_, proportion**.5, proportion**.5,
                              deciding_missing_proportion=0.2)
    def do_mnar_rows(dataset_, proportion):
        return utils.mnar_rows(dataset_, proportion**.5, proportion**.5,
                               missing_proportion_nonrandom=0.0)
    for c in [(datasets.memoize(do_mnar_rows), 'MNAR_rows'),
              (datasets.memoize(do_mar_rows), 'MAR_rows')]:
        # [(datasets.memoize(utils.mcar_total), 'MCAR_total'),
        #  (datasets.memoize(do_mcar_rows), 'MCAR_rows')]:
        for d in [.1, .3, .5]:  # if c[1] == 'MCAR_total' else [.1, .2, .3]):
            for e in ['mean_std']:
                tests_to_perform.append((b, c, d, e))
del b, c, d, e

#np.random.shuffle(tests_to_perform)

for i in range(1):
    #iter_path = "imp_bm_iter_{:d}".format(i)
    iter_path = "impute_benchmark"
    if not os.path.exists(iter_path):
        os.mkdir(iter_path)
    baseline = datasets.benchmark([
        ('Missforest_mult', datasets.memoize(
            lambda d, **_: utils.impute_missforest(d, number_imputations=10))),
        ('MICE', datasets.memoize(utils.impute_mice)),
        ('GMM', lambda log, d, full_data: missForest.impute(
            log, d, full_data, max_iterations=0,
            initial_impute=mbm.mf_initial_impute)),
        ('GP_KNN', lambda log_path, d, full_data: missForest.impute(
            log_path, d, full_data, sequential=False, print_progress=True,
            predictors=(missForest_GP.KNNGPClassification,
                        missForest_GP.KNNGPRegression),
            optimize_gp=True, use_previous_prediction=False,
            initial_impute=missForest.no_impute,
            ARD=True, n_neighbours=5, knn_type='kernel_avg', max_iterations=1)),
        ('GP_mog', lambda log_path, d, full_data: missForest.impute(
            log_path, d, full_data, sequential=False, print_progress=True,
            predictors=(missForest_GP.UncertainGPClassification,
                        missForest_GP.UncertainGPRegression),
            optimize_gp=False, use_previous_prediction=False, ARD=True,
            impute_name_replace=('GP_mog', 'GMM'),
            load_gp_model=lambda p: p.replace('GP_mog', 'GP_KNN'),
            max_iterations=1,
            initial_impute=mbm.mf_initial_impute)),
        ('mean', lambda log, d, full_data: missForest.impute(
            log, d, full_data, max_iterations=0)),
    ], dsets, tests_to_perform, do_not_compute=False, path=iter_path)

    print(baseline)
