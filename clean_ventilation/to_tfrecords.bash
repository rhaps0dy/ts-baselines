#!/bin/bash

N_HEADERS=200
set -eu # Exit if command returns 1

# This will also parse the CSV
python to_tfrecords.py number_of_categories $N_HEADERS

# Split the data set in 8
python to_tfrecords.py split_dataframe $N_HEADERS

for i in 0 1 2 3 4 5 6 7; do
	python to_tfrecords.py write_tfrecords $N_HEADERS $i &
done
wait

python to_tfrecords.py join_tfrecords_training_test_vali

