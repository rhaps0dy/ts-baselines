import numpy as np
import math
import logging

log = logging.getLogger(__name__)

def round_up(a,b):
    return ((a + b - 1) // b) * b
def round_down(a,b):
    return (a // b) * b

class TrainingManager:
    def __init__(self, talks, batch_size, bptt_len):
        self.bptt_len = bptt_len
        self.talks = talks
        self.talks_start = np.zeros([len(self.talks)], dtype=np.int32)
        self.batch_size = batch_size

        # We subtract 1 because the final word can only be used for ground truth
        self.total_talk_len = sum(len(s) for s in self.talks)
        self.total_len = round_up(self.total_talk_len - 1, self.bptt_len) + 1
        self.len_of_wholes = round_down(self.total_talk_len - 1, self.bptt_len)

        if (self.total_talk_len - 1) % self.bptt_len < (self.bptt_len / 2):
            log.warning(("The length residue is small, {:d} whereas "
                         "bptt_len={:d}").format(self.total_talk_len %
                         self.bptt_len, self.bptt_len))

        training_data_shape = [batch_size, self.total_len]
        try:
            training_data_shape += list(self.talks[0][0].shape)
        except AttributeError:
            pass
        self.training_data = np.zeros(training_data_shape, dtype=np.int32)
        seq_len = np.ones([batch_size], dtype=np.int32)
        self.sequence_length = seq_len * self.bptt_len
        self.partial_sequence_length = seq_len * ((self.total_talk_len - 1) %
                                                 self.bptt_len)
        self.new_training_permutation()
        self.epoch_steps = self.total_len // self.bptt_len

    def new_training_permutation(self):
        np.random.shuffle(self.talks)
        self.talks_start[0] = 0
        for i in range(1, len(self.talks)):
            # Negative rolls move the array to the left
            self.talks_start[i] = \
                self.talks_start[i-1] - len(self.talks[i-1])
        concat_talks = np.concatenate(self.talks)
        indices = np.random.choice(self.talks_start,
                                   size=self.batch_size % len(self.talks),
                                   replace=False)
        n_whole_talks = self.batch_size // len(self.talks)
        if n_whole_talks != 0:
            indices = np.concatenate([indices, np.tile(self.talks_start,
                                                       n_whole_talks)])
        assert indices.shape[0] == self.batch_size
        for batch_i, i in enumerate(indices):
            self.training_data[batch_i, :self.total_talk_len] = (
                np.roll(concat_talks, i, axis=0))
        return self.training_data

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i+self.bptt_len+1 > self.total_len:
            raise StopIteration
        i = self.i
        self.i += self.bptt_len
        r = self.training_data[:, i:self.i+1]
        if i >= self.len_of_wholes:
            return r, self.partial_sequence_length
        return r, self.sequence_length

class TestManager:
    def __init__(self, talks, batch_size, bptt_len):
        batch_size = min(batch_size, len(talks))
        def talks_concat(talks):
            "Greedily concatenate as many talks as necessary to have a batch"
            " of batch_size or less."
            ts = list([0, []] for _ in range(batch_size))
            for t in talks:
                min_len, min_len_i = ts[0][0], 0
                for i in range(batch_size):
                    if ts[i][0] < min_len:
                        min_len = ts[i][0]
                        min_len_i = i
                ts[min_len_i][0] += len(t)
                ts[min_len_i][1].append(t)
            talks = list(np.concatenate(t[1]) for t in ts)
            return talks

        talks = talks_concat(sorted(talks, key=lambda e: -len(e)))
        talks = sorted(talks, key=lambda e: -len(e))

        cutoffs = []
        talk_start = 0
        for i in range(len(talks)-1, -1, -1):
            while talk_start+1 < len(talks[i]):
                cutoffs.append(i+1)
                talk_start += bptt_len
        self.talk_arrays = []

        talk_start = 0
        for talk_start, cutoff in enumerate(cutoffs):
            talk_start *= bptt_len

            ts_shape = [cutoff, bptt_len+1]
            try:
                ts_shape += list(talks[0][0].shape)
            except AttributeError:
                pass
            ts = np.zeros(ts_shape, dtype=np.int32)
            tlens = np.zeros([cutoff], dtype=np.int32)
            for i in range(cutoff):
                n = tlens[i] = min(len(talks[i])-talk_start-1, bptt_len)
                ts[i,:n+1] = talks[i][talk_start:talk_start+n+1]
            self.talk_arrays.append((ts,tlens))
    def __iter__(self):
        return iter(self.talk_arrays)
