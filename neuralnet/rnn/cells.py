from abc import ABC, abstractmethod
import tensorflow as tf

__author__ = 'Alejandro González Hevia'


class BaseCellFactory(ABC):
    """Abstract base class for all cell factories."""

    @abstractmethod
    def __call__(self, num_units, activation, kernel_init, dropout_rate, layer_norm):
        pass


class LSTMCellFactory(BaseCellFactory):
    """Factory that creates variations of the LSTM cell."""

    def __call__(self, num_units, activation, kernel_init, use_dropout, keep_prob, layer_norm):
        if layer_norm:
            return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, layer_norm=True,
                                                         dropout_keep_prob=1-dropout_rate)
        else:
            lstm_cell = tf.contrib.rnn.LSTMCell(num_units, initializer=kernel_init)
            if use_dropout:
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            else:
                return lstm_cell


class GRUCellFactory(BaseCellFactory):
    """Factory that creates variations of the GRU cell."""

    def __call__(self, num_units, activation, kernel_init, use_dropout, keep_prob, layer_norm):
        gru_cell = tf.contrib.rnn.GRUCell(num_units, activation)
        if use_dropout:
            return tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)
        else:
            return gru_cell


class SimpleCellFactory(BaseCellFactory):
    """Factory that creates variations of a basic rnn cell."""

    def __call__(self, num_units, activation, kernel_init, use_dropout, keep_prob, layer_norm):
        return tf.nn.rnn_cell.BasicRNNCell(num_units)
