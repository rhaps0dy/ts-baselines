import sys, getopt
import os
import re

import tensorflow as tf

usage_str = 'python tensorflow_rename_variables.py --checkpoint=path/to/checkpoint ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(directory):
    files = os.listdir(directory)
    ckpt = re.compile(r'^(model\.ckpt-[0-9]{3,5})\..*$')
    checkpoints = set()
    for f in files:
        m = ckpt.fullmatch(f)
        if m:
            checkpoints.add(os.path.join(directory, m.group(1)))

    with tf.Session() as sess:
        for checkpoint in checkpoints:
            a_new_name = False
            names = set()
            for var_name, _ in tf.contrib.framework.list_variables(checkpoint):
                # Load the variable
                var = tf.contrib.framework.load_variable(checkpoint, var_name)

                # Set the new name
                new_name = var_name
                if not new_name.startswith('GRUD') and new_name not in {'beta1_power', 'beta2_power', 'global_step'}:
                    new_name = 'GRUD/{:s}'.format(var_name)
                    #print(var_name, 'will be renamed to', new_name)
                    var = tf.Variable(var, name=new_name)
                    a_new_name = True
                names.add(new_name)

            if 'global_step' not in names:
                print("Creating global_step")
                var = tf.Variable(int(checkpoint[checkpoint.find('-')+1:]),
                        dtype=tf.int64,
                        name='global_step')
                a_new_name = True

            if a_new_name:
                print("Doing file", checkpoint)
                saver = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                saver.save(sess, checkpoint)
            else:
                print("Skipping", checkpoint)


if __name__ == '__main__':
    rename(sys.argv[1])
