## Files to clean the MIMIC time series for use in Tensorflow

- `common.py`: functions common to all cleaning methods
- `to_tfrecords.bash`: contains the sequence of commands used to convert cleaned
MIMIC files into TFRecord files
- `to_tfrecords.py`: implements the data set splitting, counting of time steps
and joining that is used by the previous file.
- `fallback_counts_file.py`: I hardcoded some of the category counts, in case a
script failed and they were lost from RAM.
- `whiten_imputation.py`: Compute means and standard deviations of the imputed
dataset in TFRecords, to be able to normalise the inputs later.
- `unsupervised_interpolation.py`: "Interpolate" the time series of the data set
using forward imputation, recording the time since the last input as well.
