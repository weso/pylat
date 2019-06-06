import tensorflow as tf

from ..layers import BaseLayer
from ...exceptions import InvalidArgumentError
from .cells import BaseCellFactory, GRUCellFactory


class RecurrentLayer(BaseLayer):
    """Creates tensors of recurrent layers.

    This class inherits from the :obj:`BaseLayer` abstract class and provides
    an implementation to create recurrent layers that can be used in recurrent
    neural networks.

    Parameters
    ----------
    num_units : int
        Number of units that the layer will have.
    kernel_init : :obj:`callable`, optional (default='glorot_uniform')
        Function used to initialize the weights of the layer's cells.
    activation : :obj:`callable`, optional (default='relu')
        Activation function of the layer.
    dropout_rate : int, optional (default=None)
        Dropout rate of the layer.
    cell_factory : :obj:`BaseCellFactory`, optional (default=GRUCellFactory())
        Cell factory that will be used to create
        the recurrent cells of the layer.
    layer_norm : bool, optional (default=False)
        Whether layer normalization will be applied to the layer or not.
    cell_dropout : float, optional (default=0.0)
        Float value between 0 and 1 that represents the dropout rate to be
        applied to the cells of the layer.
    """

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
    """Creates tensors of bidirectional recurrent layers.

    This class is a specialization of :obj:`RecurrentLayer` which
    creates bidirectional recurrent layers.
    """

    def build_tensor(self, inputs, **kwargs):
        final_cell = self.cell_factory(self.num_units, self.activation,
                                       self.kernel_init, self.cell_dropout,
                                       self.layer_norm)
        rnn = tf.keras.layers.RNN(final_cell, name='rnn_layer')
        bidi = tf.keras.layers.Bidirectional(rnn, name='bidi_rnn')(inputs, **kwargs)
        if self.dropout_rate is not None:
            bidi = tf.keras.layers.Dropout(self.dropout_rate)(bidi, **kwargs)
        return bidi
