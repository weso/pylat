import tensorflow as tf

from ..layers import BaseLayer
from .cells import GRUCellFactory


class RecurrentLayer(BaseLayer):
    def __init__(self, num_units, kernel_init='glorot_uniform',
                 activation='relu', dropout_rate=None,
                 cell_factory=GRUCellFactory(), layer_norm=False,
                 cell_dropout=0.0):
        super().__init__(num_units, kernel_init, activation, dropout_rate)
        self.cell_factory = cell_factory
        self.layer_norm = layer_norm
        self.cell_dropout = cell_dropout

    def build_tensor(self, inputs, **kwargs):
        final_cell = self.cell_factory(self.num_units, self.activation,
                                       self.kernel_init, self.cell_dropout,
                                       self.layer_norm)
        rnn_layer = tf.keras.layers.RNN(final_cell)(inputs, **kwargs)
        if self.dropout_rate is not None:
            rnn_layer = tf.keras.layers.Dropout(self.dropout_rate)(rnn_layer,
                                                                   **kwargs)
        return rnn_layer


class BidirectionalRecurrentLayer(RecurrentLayer):
    def build_tensor(self, inputs, **kwargs):
        final_cell = self.cell_factory(self.num_units, self.activation,
                                       self.kernel_init, self.cell_dropout,
                                       self.layer_norm)
        rnn = tf.keras.layers.RNN(final_cell)
        bidi = tf.keras.layers.Bidirectional(rnn)(inputs, **kwargs)
        if self.dropout_rate is not None:
            bidi = tf.keras.layers.Dropout(self.dropout_rate)(bidi, **kwargs)
        return bidi
