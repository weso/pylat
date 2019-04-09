from src.exceptions import InvalidArgumentError
from src.neuralnet.model import BaseNeuralNetwork
from src.neuralnet.rnn.cells import GRUCellFactory
from src.neuralnet.utils import get_length_tensor, get_next_batch, \
                                he_init, last_relevant, lazy_property

import numpy as np
import tensorflow as tf

import logging
import os


class RecurrentNeuralNetwork(BaseNeuralNetwork):
    """Recurrent neural network implementation using TensorFlow.

    Parameters
    ----------
    num_epochs: int (default=100)
        Maximum number of epochs taken during the training phase.

    batch_size: int (default=200)
        Number of training instances used for each gradient update
        in the training phase.

    num_units: tuple or list of ints (default=(256,))
        Tuple containing the number of units that each recurrent layer will have. If
        the tuple contains more than one element, the recurrent layers will be stacked
        using a MultiRNNCell.

    cell_factory: BaseCellFactory object (default=GRUCellFactory())
        Instance of a cell factory that will be used to create the cells of each
        recurrent layer. For more info see :mod:`.rnn_cells`.

    activation: callable (default=None)
        Activation function applied to the output of each layer (e.g. tanh, ReLU...).

    kernel_initializer: callable (default=he_init)
        Function used to initialize the weights of the different layers.

    layer_norm: bool (default=False)
        Whether to apply layer normalization to each layer or not.

    dropout_rate: float (default=None)
        Dropout rate of each layer. Its value must be between 0 or 1.
        If its value is None or 0, no dropout is applied.

    learning_rate: float (default=1e-4)
        Learning rate used by the optimizer during the training phase.

    optimizer: tf.train.Optimizer class (default=tf.train.AdamOptimizer)
        Optimizer used during the training phase to minimize the loss
        function.

    Attributes
    ----------
    session : tf.Session
        TensorFlow session where the neural network operations will
        be executed.

    Examples
    --------
    >>> from src.neuralnet.rnn.recurrent_neural_network import \
        RecurrentNeuralNetwork
    >>> X = ['Hi, how are you doing?']
    >>> X_prep = X
    """

    def __init__(self, num_units=(256,), cell_factory=GRUCellFactory(),
                 activation=None, kernel_initializer=he_init, layer_norm=False,
                 dropout_rate=0.0, learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer, save_dir='results',
                 random_seed=42, embeddings=None):
        super().__init__(save_dir)
        self.num_units = num_units
        self.cell_factory = cell_factory
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.random_seed = random_seed
        self.w2v_embeddings = embeddings

    def init_model(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_steps = len(max(X, key=len))
        self.logger.info('Building graph. Num steps: %s', self.num_steps)
        super().init_model(X, y)

    @lazy_property
    def prediction(self):
        self.x_t = tf.placeholder(tf.int64, shape=[None, self.num_steps],
                                   name='x_input')
        self.y_t = tf.placeholder(tf.int64, shape=[None], name='y_input')

        with tf.name_scope('embeddings'):
            np_emb = np.array(self.w2v_embeddings)
            embeddings = tf.get_variable(name="W", shape=np_emb.shape, trainable=False,
                                         initializer=tf.constant_initializer(np_emb))
            rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x_t)

        with tf.name_scope('dnn'):
            logits = self._rnn(rnn_inputs, self.num_classes)
            softmax = tf.nn.softmax(logits, name='y_proba')

        self._logits = logits
        return softmax

    @lazy_property
    def loss(self):
        with tf.name_scope('loss'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_t,
                                                                  logits=self._logits,
                                                                  name='cross_entropy')
            loss_op = tf.reduce_mean(xent, name='loss')
        return loss_op

    @lazy_property
    def optimize(self):
        with tf.name_scope('train'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(self.loss)
        return train_op

    @lazy_property
    def error(self):
        with tf.name_scope('accuracy'):
            y_pred = tf.argmax(self.prediction, axis=1, name='output')
            correct = tf.equal(y_pred, self.y_t)
            accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
        return accuracy

    def _assert_valid_params(self):
        if not isinstance(self.num_units, (list, tuple)):
            raise InvalidArgumentError('num_units',
                                       'Parameter num_units must be a list or tuple with the number of units per layer')
        elif len(self.num_units) == 0:
            raise InvalidArgumentError('num_units', 'Length of num units must be greater than zero')

    def _rnn(self, inputs, num_classes):
        #use_dropout = True if self.dropout_rate == 0 else False
        #
        #if len(self.num_units) > 1:
        #    # multiple layers
        #    cells = [self.cell_factory(units, self.activation, self.kernel_initializer,
        #                               use_dropout, keep_prob, self.layer_norm)
        #             for units in self.num_units]
        #    final_cell = tf.contrib.rnn.MultiRNNCell(cells)
        #else:
        #    # single layer
        #    final_cell = self.cell_factory(self.num_units[0], self.activation, self.kernel_initializer,
        #                                   use_dropout, keep_prob, self.layer_norm)
        #rnn_output, _ = tf.nn.dynamic_rnn(final_cell, inputs,
        #                                  sequence_length=get_length_tensor(inputs),
        #                                  dtype=tf.float32)
        #last = last_relevant(rnn_output, get_length_tensor(inputs))
        #
        #with tf.name_scope('fc_layer'):
        #    fc = tf.layers.dense(last, 100, kernel_initializer=self.kernel_initializer)
        #    output = tf.layers.dense(fc, num_classes, kernel_initializer=self.kernel_initializer)
        #return output
        rnn_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, dropout=0.4))(inputs)
        
        with tf.name_scope('fc_layer'):
            fc = tf.keras.layers.Dense(100, activation='relu')(rnn_out)
            output = tf.keras.layers.Dense(num_classes)(fc)
        return output

    def __repr__(self):
        return "rnn_nc{}_ce{}_ln{}_dr{}_lr{}_op{}".format(
            self.num_units, self.cell_factory.__class__.__name__,
            self.layer_norm, self.dropout_rate, self.learning_rate,
            self.optimizer.__name__)

    def __str__(self):
        return "RNN. num_cells: {} - cell: {} \
            - norm: {} - dropout: {} - l_rate: {} - optimizer: {}".format(
            self.num_units, self.cell_factory.__class__.__name__,
            self.layer_norm, self.dropout_rate, self.learning_rate,
            self.optimizer.__name__)
