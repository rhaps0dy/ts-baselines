import numpy as np
import tensorflow as tf
from add_variable_scope import add_variable_scope
import unittest

NA_int32 = -(1 << 31)
# We can modify post_NA_int32 to allow for categories to include an extra NaN
# category. Just make it 0
post_NA_int32 = NA_int32


def _make_embedding(name, n_cats, n_dims, index, tf_float):
    "Makes embedding for a variable. Doesn't take into account missing values."
    assert n_cats > 1
    fmt_str = "{{:0{:d}b}}".format(n_dims)
    if n_dims > 1:
        initial_embeddings = np.zeros([n_cats, n_dims], dtype=tf_float.name)
        for i in range(n_cats):
            for j, c in enumerate(fmt_str.format(i)):
                if c == '1':
                    initial_embeddings[i, j] = 1.
                else:
                    initial_embeddings[i, j] = -1.
        ie = initial_embeddings
        initial_embeddings = ((ie - ie[1:].mean(axis=0, keepdims=True)) /
                              ie[1:].var(axis=0, keepdims=True))
    else:
        ie = np.arange(-(n_cats-1)/2, n_cats/2, 1.) / n_cats + 0.1
        ie = ie.astype(tf_float.name)
        initial_embeddings = ie[:, np.newaxis]
    del ie

    assert initial_embeddings.shape[0] == n_cats
    assert initial_embeddings.shape[1] == n_dims

    embeddings = tf.get_variable(
        name,
        dtype=tf_float,
        initializer=initial_embeddings,
        trainable=True)
    return tf.nn.embedding_lookup(embeddings, index)


def compress_categories(possible_cats):
    """Makes a possibly sparse array of categories, with all integers >1, of
    length `n`; into a dictionary that maps the first `n` natural numbers to
    these categories."""
    possible_cats.sort()
    rearrange_cats = [None]*possible_cats
    i = 0
    j = 0
    while i < len(possible_cats):
        c = possible_cats[i]
        while j+1 < c and i < len(possible_cats):
            rearrange_cats[j] = possible_cats[-1]
            possible_cats = possible_cats[:-1]
            j += 1
        if i < len(possible_cats):
            rearrange_cats[c] = c
        j += 1
        i += 1
    return rearrange_cats


@add_variable_scope(name='inputs')
def make_input_layer(dataset, tf_float=tf.float32):
    "make categorical inputs and concatenate them with numerical inputs"
    df, orig_cat_idx = dataset
    num_idx = list(filter(lambda k: df[k].dtype == np.float64, df.keys()))
    cat_idx = list(filter(lambda k: df[k].dtype == np.int32, df.keys()))

    n_dims_l = []
    n_cats_l = []
    add_columns_after = {}
    all_cat_maps = {}

    with tf.variable_scope('embeddings'):
        for i, k in enumerate(cat_idx.copy()):
            possible_cats = df[k].unique()
            if NA_int32 in possible_cats:
                possible_cats.remove(NA_int32)

            n_cats = len(possible_cats)
            if n_cats == 1:
                add_columns_after[k] = possible_cats[0]
                cat_idx.remove(k)
                continue

            cat_map = compress_categories(possible_cats)
            if k in orig_cat_idx:
                n_dims = int(np.ceil(np.log2(n_cats)))
            else:
                n_dims = 1
            all_cat_maps[k] = cat_map

            n_dims_l.append(n_dims)
            n_cats_l.append(n_cats)

    ret = {"num_idx": num_idx,
           "cat_idx": cat_idx,
           "n_dims_l": n_dims_l,
           "n_cats_l": n_cats_l,
           "_cat_maps": all_cat_maps,
           "_add_columns_after": add_columns_after}

    inputs_l = []
    mask_inputs_l = []

    if len(num_idx) > 0:
        ret["num_inputs"] = tf.placeholder(
            tf_float, shape=[None, len(num_idx)], name="num_ph")
        ret["num_mask_inputs"] = tf.placeholder(
            tf.bool, shape=[None, len(num_idx)], name="num_mask_ph")
        inputs_l.append(ret["num_inputs"])
        mask_inputs_l.append(ret["num_mask_inputs"])

    if len(cat_idx) > 0:
        ret["cat_inputs"] = tf.placeholder(
            tf.int32, shape=[None, len(cat_idx)], name="cat_ph")
        ret["cat_mask_inputs"] = tf.placeholder(
            tf.bool, shape=[None, len(cat_idx)], name="cat_mask_ph")

    assert len(cat_idx) == len(n_dims_l) and len(n_dims_l) == len(n_cats_l)
    for i, (k, n_dims, n_cats) in enumerate(zip(cat_idx, n_dims_l, n_cats_l)):
        input_slice = ret["cat_inputs"][:, i]
        inputs_l.append(_make_embedding(k, n_cats, n_dims, input_slice,
                                        tf_float=tf_float))
        mask = tf.stack([ret["cat_mask_inputs"][:, i]] * n_dims, axis=1)
        mask_inputs_l.append(mask)

    assert len(inputs_l) > 0, "The dataframe to get inputs from is empty!"
    ret["inputs"] = tf.concat(inputs_l, axis=1)
    ret["mask_inputs"] = tf.concat(mask_inputs_l, axis=1)
    assert ret["inputs"].get_shape()[1] == ret["mask_inputs"].get_shape()[1]
    return ret


def preprocess_dataframe(dataframe, input_layer):
    """Makes categories zero-based and contiguous. Removes categories with one
    value."""
    for c, m in input_layer["_cat_maps"].items():
        inv_m = dict(zip(m, range(len(m))))
        inv_m[NA_int32] = post_NA_int32
        dataframe[c] = dataframe[c].map(inv_m).astype(np.int32)

        assert sorted(dataframe[c].unique()) == \
            sorted([post_NA_int32] + list(range(1, len(m)+1)))
    return dataframe


def postprocess_dataframe(dataframe, input_layer):
    """Restores sparse 1-based categories. Re-adds categories with one value."""
    for c, m in input_layer["_cat_maps"].items():
        dataframe[c] = dataframe[c].map(m.__getitem__).astype(np.int32)
    return dataframe


class TestCompressCategories(unittest.TestCase):
    def test_compress(self):
        possible_cats = list(range(1, 6))
        self.assertEqual(compress_categories(possible_cats),
                         dict(map(lambda x: (x, x), possible_cats)))
        possible_cats[2] = 7
        self.assertEqual(compress_categories(possible_cats), [1, 2, 7, 4, 5])
        self.assertEqual(compress_categories([2, 5, 8]), [8, 2, 5])
        self.assertEqual(compress_categories([2, 10, 11]), [11, 2, 10])

    def test_random_compress(self):
        for _ in range(100):
            n = np.random.randint(1, 31)
            cats = np.random.choice(n * 4, size=n, replace=False)
            cats += 1  # ensure they start at 1
            l = compress_categories(cats)
            self.assertEqual(set(l), set(cats))
