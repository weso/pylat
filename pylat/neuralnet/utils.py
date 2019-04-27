from gensim.models import KeyedVectors
from .embeddings import Word2VecEmbedding
from pathlib import Path

import functools
import os
import tensorflow as tf

he_init = tf.contrib.layers.variance_scaling_initializer()


def get_next_batch(x, y, batch_idx, batch_size):
    """ Fetch next batch from data set for training.

    :param x: Feature matrix of size (num_samples, num_features).
    :param y: Label matrix of size (num_samples, 1)
    :param batch_idx: Index of the current batch.
    :param batch_size: Size of the batch
    :return: Tuple containing the current x and y batch.
    """
    start = batch_idx * batch_size
    end = (batch_idx + 1) * batch_size
    end = min(end, len(x) + 1)
    return x[start:end], y[start:end]


def last_relevant(output, length):
    """ Get relevant data from the output of a dynamic RNN layer

    :param output: Output tensor of the RNN layer.
    :param length: Length of the inputs of the layer.
    :return: Tensor containing the relevant output data for each data instance.
    """
    batch_size = tf.shape(output)[0]
    rows_idx = tf.range(0, batch_size)
    col_idx = tf.maximum(1, length) - 1
    final_idx = tf.stack([rows_idx, col_idx], axis=1)
    relevant = tf.gather_nd(output, final_idx)
    return relevant


def get_length_tensor(sequence):
    """ Return a tensor that computes the length of a sequence.

    This function can be used when working with padded textual
    data. It computes the actual length of each text input,
    without taken into account the padded elements.
    :param sequence: Tensor containing the input data.
    :return: Tensor that computes the actual length of the data.
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    return tf.cast(length, tf.int32)


def lazy_property(func):
    attribute = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)

    return decorator


def convert_vec_embedding(input_file, output_dir):
    """Converts a .vec file into the format used by the Embedding classes.

    Parameters
    ----------
    input_file
    output_dir

    Returns
    -------

    """
    path = Path(input_file)
    file_name = path.stem
    model = KeyedVectors.load_word2vec_format(input_file, unicode_errors='ignore')
    embedding = Word2VecEmbedding(model=model)
    embedding.save_embeddings(os.path.join(output_dir, file_name))
