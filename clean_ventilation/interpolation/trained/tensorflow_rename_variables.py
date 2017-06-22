import sys, getopt
import os
import re

import pickle_utils as pu
import tensorflow as tf

def rename(checkpoint):
    with tf.Session() as sess:
        a_new_name = False
        names = set()
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint):
            # Load the variable
            # Set the new name
            new_name = var_name
            if var_name.startswith('Bayes') and 'Adam' not in var_name:
                var = tf.contrib.framework.load_variable(checkpoint, var_name)
            #    new_name = 'BayesDropout/bayes_dropout_cell/num_inputs/{:s}'.format(var_name)
            #    print(var_name, 'will be renamed to', new_name)
                var = tf.Variable(var, name=new_name.replace("BayesDropout", "BayesDropout/rnn"))
            #    a_new_name = True
            #names.add(new_name)

        #if 'global_step' not in names:
        #    print("Creating global_step")
        #    var = tf.Variable(0, dtype=tf.int64, name='global_step')

        #if a_new_name:
        print("Doing file", checkpoint)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, checkpoint)
        #else:
        #    print("Skipping", checkpoint)


if __name__ == '__main__':
    n = pu.load('validated_best.pkl')
    checkpoint = 'new-ckpt-{:d}'.format(n)
    rename(checkpoint)
