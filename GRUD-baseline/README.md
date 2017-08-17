# GRUD-baseline layout

## Pre-imputation used by the Bayesian missing-data dropout RNN
  
- `fast_smooth_category_counts.py`: More complete implementation of the
  `fast_smooth_category_counts.ipynb` notebook in the parent folder. It learns
  the optimal smoothing parameter for counts. Used as pre-imputation for the
  `BayesDropout` model.
  
- `bb_alpha_inputs.py`: Learn a Bayesian Neural Network
  using
  [Black-box alpha-divergence minimization](https://arxiv.org/abs/1511.03243),
  equations explicit
  in
  [Learning and Policy Search in Stochastic Dynamical Systems with Bayesian Neural Networks](https://arxiv.org/abs/1605.07127),
  that predicts the value at the next step of the time series, for all inputs.
  
- `bb_alpha_all_inputs.py`: the same as the previous one, but using the official
  "good" input system of Tensorflow (which turns out to be bad and slow, I
  hypothesize that that's because it spends a lot of time context-switching
  between useless threads). Thus it can directly read the files cleaned by
  `../clean_ventilation/to_tfrecords.bash`
  
## Read input data
	
- `imputation_read_tfrecords.py`: read the files prepared for imputation by
  `../clean_ventilation/datasets_for_imputation.bash`. These will have, for each
  feature of the time-series, all the transitions of that feature, disassociated
  of their patients or the features before and after them. This is used by
  `bb_alpha_inputs_all.py`.
  
- `read_tfrecords.py`: read the time series from TFRecords files created by
  `../clean-ventilation.py/to_tfrecords.bash`
  
## RNN models

- `gru_ln_dropout_cell.py`: Implementation
  of [GRU-D RNN](https://arxiv.org/abs/1606.01865) cell, with two different
  kinds of RNN dropout:
  - Variational dropout [(Gal and Ghahramani, 2015)](https://arxiv.org/abs/1512.05287)
  - Dropout at the gate, without memory
    loss [(Semeniuta et al., 2016)](https://arxiv.org/abs/1603.05118)

- `model.py`: Implementations of different prediction models. They are all put
  here with the intention that, when doing experiments, one can choose between
  them by passing a flag to `train.py`. The models are:
  - GRU-D, with gated dropout and embedded categories (function `GRUD`)
  - Bayesian dropout RNN (function `BayesDropout`). This would take samples from
    the input predictions learned by `bb_alpha_inputs_all.py`, and then train a
    normal GRU RNN with variational dropout.
  
- `train.py`: code for training, validation (measuring for hyperparameter
  optimisation) and testing the time-series-imputation RNN.

## Miscellaneous

- `retroactive_measure.py`: compute the validation metrics of previously saved
  checkpoints of a neural network. For some of the models, this is expensive, so
  we start by computing the performance of every 2^k checkpoints, then every
  2^(k-1) checkpoints, etc... This way, we can interrupt the testing whenever,
  and still have a reasonably full picture of what the model's performance was
  as training time passed.

- `requirements.txt`: the list of Python packages needed to run these files.

- `tensorflow_rename_variables.py`: Change the name of variables in a saved
  checkpoint. Sometimes when changing the code in `model.py` or `train.py`, the
  path of the variable to load would change. This is a problem when loading
  input layers trained by `bb_alpha_inputs_all.py`, for example.

- `GP_infer.py`: Attempts at using the [GPflow](http://github.com/gpflow/gpflow)
  library for GP regression. The aim was to use it to predict the next
  time-series value for tie-series imputation.


## Directories

- `plot_aggregate_metrics`: read from the Tensorflow summaries
  (`plot_aggregate_metrics/read_all_logs.py`) the calculated validation metrics,
  and plot them (`plot_aggregate_metrics/plot.py`)
  
- `with_feed_dict`: model and training routine using `feed_dict` and tensorflow
  Placeholders instead of tensorflow Queues for input. Not really used.
  
- `bptt_utils`: utility classes for doing Truncated Backpropagation Through
  Time. Not really used either; the time series in MIMIC are short enough to be
  fully backpropagated (at most ~2000 effective time steps)
