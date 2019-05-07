from abc import ABC, abstractmethod
import tensorflow as tf

__author__ = 'Alejandro González Hevia'


class BaseCellFactory(ABC):
    """Abstract base class for all cell factories."""

    @abstractmethod
    def __call__(self, num_units, activation, kernel_init, dropout, layer_norm):
        pass


class LSTMCellFactory(BaseCellFactory):
    """Factory that creates variations of the LSTM cell."""

    def __call__(self, num_units, activation, kernel_init, dropout, layer_norm):
            return tf.keras.layers.LSTMCell(num_units, dropout=dropout,
                                            name='lstm')


class GRUCellFactory(BaseCellFactory):
    """Factory that creates variations of the GRU cell."""

    def __call__(self, num_units, activation, kernel_init, dropout, layer_norm):
        return tf.keras.layers.GRUCell(num_units, activation,
                                       kernel_initializer=kernel_init,
                                       dropout=dropout, name='gru')


class SimpleCellFactory(BaseCellFactory):
    """Factory that creates variations of a basic rnn cell."""

    def __call__(self, num_units, activation, kernel_init, dropout, layer_norm):
        return tf.keras.layers.SimpleRNNCell(num_units, activation,
                                             kernel_initializer=kernel_init,
                                             dropout=dropout,
                                             name='simple_cell')
