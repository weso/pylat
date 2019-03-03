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
