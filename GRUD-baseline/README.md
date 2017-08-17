## GRUD-baseline layout

- `baseline.py`: compute a classification baseline for the MIMIC time series
  ventilation-outcome task. It uses only static features of each patient,
  including some natural language features found in the "diagnosis" written by
  the person at the emergency room entrance desk.
  
- `GP_infer.py`: Attempts at using the [GPflow](http://github.com/gpflow/gpflow)
  library for GP regression. The aim was to use it to predict the next
  time-series value for tie-series imputation.
  
- `fast_smooth_category_counts.py`: More complete implementation of the
  `fast_smooth_category_counts.ipynb` notebook in the parent folder. It learns
  the optimal smoothing parameter for counts.
  
- `bb_alpha_inputs.py`: Learn a Bayesian Neural Network
  using
  [Black-box alpha-divergence minimization](https://arxiv.org/abs/1511.03243),
  equations explicit
  in
  [Learning and Policy Search in Stochastic Dynamical Systems with Bayesian Neural Networks](https://arxiv.org/abs/1605.07127),
  that predicts the value at the next step of the time series, for all inputs.
  
- `bb_alpha_inputs_all.py`: the same as the previous one, but using the official
  "good" input system of Tensorflow (which turns out to be bad and slow, I
  hypothesize that that's because it spends a lot of time context-switching
  between useless threads). Thus it can directly read the files cleaned by
  `../clean_ventilation/to_tfrecords.bash`

- `gru_ln_dropout_cell.py`: Implementation
  of [GRU-D RNN](https://arxiv.org/abs/1606.01865) cell, with two different
  kinds of RNN dropout:
  - Variational dropout [(Gal and Ghahramani, 2015)](https://arxiv.org/abs/1512.05287)
  - Dropout at the gate, without memory
    loss [(Semeniuta et al., 2016)](https://arxiv.org/abs/1603.05118)
	
- `imputation_read_tfrecords.py`: read the files prepared for imputation by
  `../clean_ventilation/datasets_for_imputation.bash`. This is used by
  `bb_alpha_inputs_all.py`.
