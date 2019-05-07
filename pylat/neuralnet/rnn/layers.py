import tensorflow as tf

from ..layers import BaseLayer
from ...exceptions import InvalidArgumentError
from .cells import BaseCellFactory, GRUCellFactory


class RecurrentLayer(BaseLayer):
    def __init__(self, num_units, kernel_init='glorot_uniform',
                 activation='relu', dropout_rate=None,
                 cell_factory=GRUCellFactory(), layer_norm=False,
                 cell_dropout=0.0):
        super().__init__(num_units, kernel_init, activation, dropout_rate)
        self.cell_factory = cell_factory
        self.layer_norm = layer_norm
        self.cell_dropout = cell_dropout
        self._check_valid_cell()

    def build_tensor(self, inputs, **kwargs):
        final_cell = self.cell_factory(self.num_units, self.activation,
                                       self.kernel_init, self.cell_dropout,
                                       self.layer_norm)
        rnn_layer = tf.keras.layers.RNN(final_cell, name='rnn_layer')(inputs, **kwargs)
        if self.dropout_rate is not None:
            rnn_layer = tf.keras.layers.Dropout(self.dropout_rate)(rnn_layer,
                                                                   **kwargs)
        return rnn_layer

    def _check_valid_cell(self):
        if not isinstance(self.cell_factory, BaseCellFactory):
            raise InvalidArgumentError('cell',
                                       'Cell factory must be an instance '
                                       'of a subclass of BaseCellFactory.')


class BidirectionalRecurrentLayer(RecurrentLayer):
    def build_tensor(self, inputs, **kwargs):
        final_cell = self.cell_factory(self.num_units, self.activation,
                                       self.kernel_init, self.cell_dropout,
                                       self.layer_norm)
        rnn = tf.keras.layers.RNN(final_cell, name='rnn_layer')
        bidi = tf.keras.layers.Bidirectional(rnn, name='bidi_rnn')(inputs, **kwargs)
        if self.dropout_rate is not None:
            bidi = tf.keras.layers.Dropout(self.dropout_rate)(bidi, **kwargs)
        return bidi
