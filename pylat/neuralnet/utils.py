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


def convert_vec_embedding(input_file, output_dir):
    """Converts a .vec file into the format used by the Embedding classes.

    This method converts a .vec file into two different files used by the
    WordEmbedding classes. The first one is a .npy file that holds the
    weights of each token. The second one is a .vocab file that contains
    the ordered list of words contained in the embedding.

    Parameters
    ----------
    input_file : str
        Path of the .vec file to be converted.

    output_dir : str
        Directory where the new converted files will be saved.
    """
    path = Path(input_file)
    file_name = path.stem
    model = KeyedVectors.load_word2vec_format(input_file, unicode_errors='ignore')
    embedding = Word2VecEmbedding(model=model)
    embedding.save_embeddings(os.path.join(output_dir, file_name))
