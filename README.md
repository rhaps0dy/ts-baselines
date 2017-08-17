This repository is best viewed on a Git web interface that supports markdown
README files, such as [Gogs](https://gogs.io/), or the proprietary GitLab,
GitHub, and BitBucket.

# ts-baselines
This repository contains most of the code related to my MSc dissertation. The
exception is the code used to clean the MIMIC ICU dataset. The name stands for
"Time Series baselines", and it originally was just that.

It also used to contain some of Qixuan "Tony" Feng's earliest attempts at his
own project, since we thought we could share a lot of boilerplate code, but this
turned out to not be the case. His files are in another branch of the
repository, called `qixuan-project`.

# Layout

- `/clean_ventilation`: This directory contains code used to take the cleaned
  full MIMIC data set and feed it into RNNs implemented in Tensorflow.
- `/GRUD-baseline`: This contains code to
  compute [Che et al.'s GRU-D model](https://arxiv.org/abs/1606.01865) for time
  series with missing data, and attempts by me to improve it. It also contains
  an implementation of a feed-forward Bayesian neural network using
  [Black-box alpha-divergence minimization](https://arxiv.org/abs/1511.03243).
- `/timeless-imputation`: contains code related to imputation in static
  data-sets. There is an implementation
  of [Self-normalising Neural Networks](https://arxiv.org/abs/1706.02515),
  of
  [MissForest](https://academic.oup.com/bioinformatics/article/28/1/112/219101/MissForest-non-parametric-missing-value-imputation) (although
  for the tests in the dissertation their implementation is used) and of my own
  novel methods:
  - a Variational Bayesian Gaussian Mixture Model implementation based
    on
    [Scikit-learn's](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture). 
  - A general framework for turning machine learning models into imputation
    methods.
  - Implementations of several kernels for Gaussian Process regression and
    classification that are aware of missing values, for use
    with [SheffieldML's GPy library](http://gpy.readthedocs.io/en/deploy/).
	
## Files:
  - `smooth_category_counts.ipynb`: Smoothing the counts of a categorical value
    at a time step for every series, across different time steps. The
    smoothing kernel centered at $t'$ is of form $e^-\alpha |(t - t')|$, and
    $\alpha$ is optimized to offer maximum likelihood of the data.
  - `fast_smooth_category_counts.ipynb`: same but faster.
  - `feature_extraction.py`: attempt at extracting features from time series
    using [tsfresh](https://github.com/blue-yonder/tsfresh) to use feature
    selection algorithms on the resulting static data set.
  - `Plot GMM cluster weights and max(2, 2y) function.ipynb`: What are the
    learned cluster weights, and how does that function look? These questions
    explored here.

# Instructions for doing time-series classification

* To clean MIMIC data-set for ventilations: run `clean_ventilation/to_tfrecords.bash`

* To input the dataset to GRUD-baseline: 
There are two things required: TFRecord files with the data, and .pkl.gz files
that tell you how to interpret it. These should be put in a
folder, that will be passed as the `--dataset` flag.

* Other notes about the models on GRUD-baseline
  * I split the training, test and validation sets randomly. Thus, sometimes it
    happens that a feature can have several categories, but it has none in the
    training set.
