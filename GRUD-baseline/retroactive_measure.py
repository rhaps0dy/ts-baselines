import pickle_utils as pu
import os.path
import os
import sys
import re
import math

@pu.memoize("{0:s}/checkpoints.pkl")
def get_checkpoints(directory):
    files = os.listdir(directory)
    ckpt = re.compile(r'^(model\.ckpt-[0-9]{3,5})\..*$')
    checkpoints = set()
    for f in files:
        m = ckpt.fullmatch(f)
        if m:
            checkpoints.add(os.path.join(directory, m.group(1)))
    checkpoints = sorted(list(checkpoints))
    return checkpoints, set()

def get_next_pow2(n):
    i = 1
    while i < n:
        i *= 2
    return i

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: {:s} <folder> <extra args>".format(sys.argv[0]))
        sys.exit(1)
    ckpts, seen = get_checkpoints(sys.argv[1])
    limit = get_next_pow2(len(ckpts))
    step = limit//2

    def do_checkpoint(j):
        if ckpts[j] not in seen:
            seen.add(ckpts[j])
            pu.dump((ckpts, seen),
                    "{:s}/checkpoints.pkl".format(sys.argv[1]))
            print("Doing checkpoint", ckpts[j])
            os.system(("python train.py --command=validate --num_epochs=1 "
                      "--batch_size=256 '--load_file={:s}' '--log_dir={:s}' {:s}")
                      .format(ckpts[j], sys.argv[1], sys.argv[2]))

    do_checkpoint(-1)
    while step >= 1:
        for i in range(0, limit, step):
            j = int(round(i / limit * len(ckpts)))
            do_checkpoint(j)
        step //= 2
