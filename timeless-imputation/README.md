There are a lot of files in this directory, in this list they are ordered by
importance. Most important go first.

## Gaussian Process imputation
- `missForest.py`: Code for Python implementation of missForest, or for any method that uses:
  - An initial imputation method
  - Subsequent rounds of using classifiers and regressors to improve the
  imputation
  
  missForest uses zero imputation for the first, and Random Forests
  for the second step. My Mixture of Gaussians uses itself as the initial
  imputation method, and then sets the maximum nubmer of iterations to 0.
  Many examples of how to use this can be seen in the files in the "Running and
  Joining Experiments" subsection.
  
- `missForest_GP.py`: wrappers over different kinds of Gaussian Processes
  suitable for use with `missForest.py`.
- `mog_rbf.py`: implentation of the RBF kernel with partially observed points as
  inputs. The unobserved dimensions of the points follow either a Gaussian or a
  Mixture of Gaussians distribution.
- `mog_rbf_tf.py`: re-implementation of `mog_rbf.py` in Tensorflow, hoping that
  the graphics card acceleration would make it faster, and the automatic
  differentiation would allow us to compute gradients and estimate gradient
  hyperparameters in reasonable time. This was a false hope.
- `knn_kernel.py`: implementation of RBF + White noise kernel with kernel
  K-nearest neighbours imputation for missing values.


## Gaussian Mixture Model imputation
- `gmm_impute.py`: contains routines to calculate a single multivariate
  Gaussian's pdf at a partially observed point, to draw samples from a Mixture
  of Gaussians, calculate the conditional p(Xmis | Xobs), and some more.
- `missing_bayesian_mixture.py`: implements the Missing Bayesian GMM model based
  on the Scikit-learn BayesianMixture class.
- `sklearn_base.py`: Patch to scikit-learn's `sklearn/mixture/base.py` file that
  allows us to pass missing values to the `BayesianMixtureMissingData`.

## Measuring performance
- `datasets.py`: Function to load the list of benchmarking datasets (from UCI,
  via the `mlbench` R package), and function to perform comprehensive benchmarks.
- `utils.py`: Functions to "ampute" (opposite of impute) datasets according to
  different missing mechanisms, to convert dataframes from R to Python and vice
  versa, and to calculate NRMSE, PFC and log-likelihood.

- `test_nth_iteration.py`: test a missForest-like method's performance at the
  nth iteration, by passing the path of its saved file as a command line
  argument, retroactively by loading the checkpoints from disk.
- `iteration_measurer.py`: performance measures for every iteration of the
  missForest-like-method.

## Other attempts at solving the problem
- `vae.py`: implementation of the [Variational Autoencoder](https://arxiv.org/abs/1312.6114)
- `category_dae.py`: Categorical embeddings for a Denoising Autoencoder.
- `denoising_ae.py`: My own idea to make an auto-encoder NN to impute by
  replacing missing values with existing ones.
- `selu.py`: Copy of
  the [code by Guenter Klambauer](https://github.com/bioinf-jku/SNNs) to
  implement Self-normalising Neural Networks in Tensorflow (Klambauer et al.
  2017)
- `neural_networks.py`:  Implementation of a Denoising Autoencoder for
  imputation [Gondara and Wang, 2017](https://arxiv.org/abs/1705.02737). With
  this we found out that their algorithm does not perform very well.
- `gmm_dae.py`: Denoising Autoencoder that starts from a Gaussian Mixture Model

  
## Running and joining experiments
These files are very similar to each other and contain slight variations of
running experiments by calling `datasets.benchmark` (defined in `datasets.py`).
Some of them set the `do_not_compute` flag which measures only the imputation
methods that have saved their results to disk.
`collect_all.py`, `collect_experiment_battery.py`, `combine_experiments.py`,
`experiment_battery.py`, `experiment_prior.py`

- `Plot report.ipynb`: Presents results obtained using some of the files above
  in a readable way.

- `Algorithm benchmarks.ipynb`: Display the performance of experiments, using
  the `do_not_compute` flag.
- `[stone] Algorithm benchmarks.ipynb`: Same but for another computer; having a
  file synchronizer forces you to do this with iPython notebooks.

## Debugging and failed tests
- `Debugging BGMM.ipynb`: notebook to figure out what was wrong with the
   implementation of theBayesian Gaussian Mixture Model
- `plot_gmm_covariances.ipynb`: Plotting the means and covariances of a GMM in a
  2-dimensional data set, also for debugging.
- `BayesianMixture approximation type.ipynb`: trying to learn a Gaussian Mixture
  Model on an explicit polynomial space. Didn't work very well.
- `GP Multiclass.ipynb`: attempting to use a multi-class Gaussian Process. It's
  not very easy in `GPy`.
- `Gaussian Process vs Random Forests.ipynb`: Testing to know what it takes to
  make a GP perform better than a RF at regression.
- `Ionosphere dataset.ipynb`: Short test of three kinds of imputation
- `Pulsar dataset imputation.ipynb`: Imputation of the HTRU_2 dataset, again trying things
- `Regressor faceoff.ipynb`: Testing performance of different regression methods
  to predict one of the variables in BostonHousing based on the others.
- `Classifier faceoff.ipynb`: same but with classifiers.
- `[stone] Regressor faceoff.ipynb`: Same but for another computer; iPython
  notebooks are modified when running them so having a Dropbox-like file
  synchronizer forces you to do this.

## Setting up environment and experiments in new machines
These files are used to run experiments in different computers.

- `prepare_experiments.bash`: setting up the environment for experiments in a
  fresh CS Department computer.
- `requirements.txt`: list of the Python packages needed to run this code
- `install.R`: R script to install the needed packages

## Miscellaneous
- `add_variable_scope.py`: useful Python decorator to create a Tensorflow
  variable scope around a function.
- `bb_alpha_inputs.py`: symlink to the file of the same name in `../GRUD-baseline`
