import sys
assert len(sys.argv) == 3, "need to specify the directory from which to take" \
    "and the destination file"

import missForest
import datasets
import itertools
import utils
import pickle_utils as pu
import missing_bayesian_mixture as mbm
import missForest_GP

_ds = datasets.datasets()
dsets = list((x, _ds[x]) for x in ["BostonHousing", "Ionosphere"])

algos = [('GMM', lambda log, d, full_data: missForest.impute(
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
         ('GP_KNN_prior', lambda log_path, d, full_data: missForest.impute(
        log_path, d, full_data, sequential=False, print_progress=True,
        predictors=(missForest_GP.KNNGPClassification,
                    missForest_GP.KNNGPRegression),
        optimize_gp=True, use_previous_prediction=False,
        initial_impute=missForest.no_impute,
        ARD=True, n_neighbours=5, knn_type='kernel_avg', max_iterations=1,
        use_informed_prior=True)),
    ('GP_mog_prior', lambda log_path, d, full_data: missForest.impute(
        log_path, d, full_data, sequential=False, print_progress=True,
        predictors=(missForest_GP.UncertainGPClassification,
                    missForest_GP.UncertainGPRegression),
        optimize_gp=False, use_previous_prediction=False, ARD=True,
        impute_name_replace=('GP_mog_prior', 'GMM'),
        load_gp_model=lambda p: p.replace('GP_mog_prior', 'GP_KNN_prior'),
        max_iterations=1,
        initial_impute=mbm.mf_initial_impute,
        use_informed_prior=True)),
        ("mean", missForest.impute),
        ('MICE', datasets.memoize(utils.impute_mice)),
        ('Missforest_mult', datasets.memoize(
            lambda d, **_: utils.impute_missforest(d, number_imputations=10))),
        ]


def do_mcar_rows(dataset_, proportion):
    return utils.mcar_rows(dataset_, proportion**.5, proportion**.5)
def do_mar_rows(dataset_, proportion):
    return utils.mar_rows(dataset_, proportion**.5, proportion**.5,
                          deciding_missing_proportion=0.2)
def do_mnar_rows(dataset_, proportion):
    return utils.mnar_rows(dataset_, proportion**.5, proportion**.5,
                           missing_proportion_nonrandom=0.0)
def do_mnar_r_rows(dataset_, proportion):
    missing_proportion_columns = proportion**.5 * 1.4
    if missing_proportion_columns > 100.:
        missing_proportion_columns = 50
    else:
        missing_proportion_columns = proportion**.5 / 1.4
    return utils.mnar_random_rows(dataset_, proportion**.5,
                                  missing_proportion_columns,
                                  proportion**.5 / 1.4,
                                  missing_proportion_nonrandom=0.0)

ampute_funs = [(datasets.memoize(do_mnar_rows), 'MNAR_rows'),
               (datasets.memoize(do_mnar_r_rows), 'MNAR_R_rows'),
               (datasets.memoize(do_mar_rows), 'MAR_rows'),
               (datasets.memoize(utils.mcar_total), 'MCAR_total'),
               (datasets.memoize(do_mcar_rows), 'MCAR_rows')]

tests_to_perform = []
for b in dsets:
    for c in ampute_funs:
        for d in [.1, .3, .5, .7, .9]:
            tests_to_perform.append((b, c, d, "mean_std"))

baseline = datasets.benchmark(algos, dsets, tests_to_perform, do_not_compute=True,
                              path=sys.argv[1])
print(baseline)
pu.dump(baseline, sys.argv[2])
