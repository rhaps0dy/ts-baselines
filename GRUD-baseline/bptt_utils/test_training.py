import unittest
import numpy as np
import math

from training import TrainingManager, TestManager

class TrainingManagerTest(unittest.TestCase):
    TestedClass = TrainingManager

    @classmethod
    def example_list_from(klass, talks, bptt_len):
        tm = klass.TestedClass(talks, 1, bptt_len)
        return list(list(a[0,:b[0]+1]) for a, b in tm)

    def test_good_iteration(self):
        talks = list(np.arange(i) for i in [12, 4, 7])
        tm = self.TestedClass(talks, 10, 4)
        examples = list(tm)
        self.assertFalse(examples[0][0][:,0].any(), "All the rows start with a zero")
        self.assertTrue(all((exm[0][0].shape[0] == 5) for exm in examples),
                        "All examples need to be bptt_len + 1 in length")

    def test_multidimensional_input(self):
        talks = list(np.stack(list(np.repeat(j, 3) for j in range(i)), axis=0)
                     for i in [10,5,2])
        tm = self.TestedClass(talks, 10, 4)
        examples = list(tm)
        self.assertFalse(examples[0][0][:,0,:].any(), "All the rows start with a zero")
        self.assertTrue(all((exm[0][0].shape[0] == 5) for exm in examples),
                        "All examples need to be bptt_len + 1 in length")

    def test_off_by_one(self):
        exms = self.example_list_from([np.arange(4)], 3)
        self.assertEqual(exms, [[0,1,2,3]])
        exms = self.example_list_from([np.arange(5)], 3)
        self.assertEqual(exms, [[0,1,2,3], [3,4]])
        exms = self.example_list_from([np.arange(6)], 3)
        self.assertEqual(exms, [[0,1,2,3], [3,4,5]])
        exms = self.example_list_from([np.arange(7)], 3)
        self.assertEqual(exms, [[0,1,2,3], [3,4,5,6]])
        exms = self.example_list_from([np.arange(8)], 3)
        self.assertEqual(exms, [[0,1,2,3], [3,4,5,6], [6,7]])

    def test_all_lengths(self):
        bptt_len = 10
        for conversation_len in range(3, bptt_len*3+2):
            tm = self.TestedClass([np.arange(conversation_len)], 1, bptt_len)
            tm = list(tm)
            self.assertTrue(all(e.shape[1]==(bptt_len+1) for e, l in tm), "All"
                            " examples must have the appropriate array size")
            self.assertEqual(len(tm), int(math.ceil((conversation_len-1)/bptt_len)),
                             "Enough elements for all the examples")
            for i in range(len(tm)-1):
                self.assertEqual(tm[i+1][0].shape, tm[0][0].shape)
                for j in range(tm[0][0].shape[0]):
                    self.assertEqual(tm[i][0][j,-1], tm[i+1][0][j, 0],
                                    "Sentences are a continuation")

class TestManagerTest(TrainingManagerTest):
    TestedClass = TestManager

    def test_all_lengths(self):
        "Overriden"
        bptt_len = 10
        max_conversation_len = bptt_len*3+2
        for conversation_len in range(3, max_conversation_len+1):
            _tm = self.TestedClass([np.arange(conversation_len)], 1, bptt_len)
            tm = list(_tm)
            self.assertTrue(all(e.shape[1]==(bptt_len+1) for e, l in tm), "All"
                            " examples must have the appropriate array size")
            self.assertEqual(len(tm), int(math.ceil((conversation_len-1)/bptt_len)),
                             "Enough elements for all the examples")
            for i in range(len(tm)-1):
                len_cur = tm[i][0].shape[0]
                len_next = tm[i+1][0].shape[0]
                self.assertLessEqual(len_next, len_cur)
                for j in range(len_next):
                    self.assertEqual(tm[i][0][j,-1],
                        tm[i+1][0][j, 0], "Sentences are a continuation")

    def test_all_lengths_more_batch(self):
        bptt_len = 10
        max_conversation_len = bptt_len*3+2
        talks = list(np.arange(i) for i in range(3,max_conversation_len+1))
        _tm = self.TestedClass(talks, len(talks), bptt_len)
        tm = list(_tm)
        self.assertTrue(all(e.shape[1]==(bptt_len+1) for e, l in tm), "All"
                        " examples must have the appropriate array size")
        self.assertEqual(len(tm), int(math.ceil((max_conversation_len-1)/bptt_len)),
                            "Enough elements for all the examples")
        for i in range(len(tm)-1):
            len_cur = tm[i][0].shape[0]
            len_next = tm[i+1][0].shape[0]
            self.assertLessEqual(len_next, len_cur)
            for j in range(len_next):
                self.assertEqual(tm[i][0][j, -1], tm[i+1][0][j, 0],
                                "Sentences are a continuation")
        # We subtract 1 because conversation_len-1 is the number of words with
        # true label of a conversation

if __name__ == '__main__':
    unittest.main()
