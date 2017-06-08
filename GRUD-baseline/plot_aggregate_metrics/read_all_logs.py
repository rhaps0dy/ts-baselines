import tensorflow as tf
import os
import re
import os.path
import sys
import collections
import pickle_utils as pu

tfevents = re.compile(r'^events\.out\.tfevents\.[0-9]+\.[a-z]+$')

def read_all_metrics():
    metrics = collections.defaultdict(lambda: {}, {})
    for f in os.listdir(sys.argv[1]):
        if not tfevents.fullmatch(f):
            continue
        for e in tf.train.summary_iterator(os.path.join(sys.argv[1], f)):
            for v in e.summary.value:
                if v.tag.startswith("metrics/"):
                    metrics[v.tag[8:]][e.step] = v.simple_value
    return dict(metrics)

if __name__ == '__main__':
    pu.dump(read_all_metrics(), 'metrics.pkl.gz')
