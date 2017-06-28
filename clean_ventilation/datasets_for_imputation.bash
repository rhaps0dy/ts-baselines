#!/bin/bash

export CUDA_VISIBLE_DEVICES=""
python unsupervised_interpolation.py --command=AddInterpolationInputs --dataset=dataset/validation_0.tfrecords &
python unsupervised_interpolation.py --command=AddInterpolationInputs --dataset=dataset/train_0.tfrecords
python whiten_imputation.py --command=Means --dataset=dataset/train_0.tfrecords-imputation
python whiten_imputation.py --command=Stddevs --dataset=dataset/train_0.tfrecords-imputation
