# ts-baselines
Time Series baselines

## Cleaning MIMIC data-set for ventilations

run `clean_ventilation/to_tfrecords.bash`

## Input dataset to GRUD-baseline

There are two things required: TFRecord files with the data, and .pkl.gz files
that tell you how to interpret it. These should be put in a
folder, that will be passed as the `--dataset` flag.

TODO: complete section

## Notes about the models on GRUD-baseline

 * I split the training, test and validation sets randomly. Thus, sometimes it
   happens that a feature can have several categories, but it has none in the
   training set. Maybe this is a problem or maybe not.
