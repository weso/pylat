from easynlp.exceptions import InvalidArgumentError
from ..model import BaseNeuralNetwork

import numpy as np
import tensorflow as tf

import logging


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
    >>> from easynlp.neuralnet.rnn.recurrent_neural_network import \
        RecurrentNeuralNetwork
    >>> X = ['Hi, how are you doing?']
    >>> X_prep = X
    """

    def __init__(self,  rnn_layers, fc_layers, learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer, save_dir='results',
                 random_seed=42, embeddings=None):
        super().__init__(save_dir)
        self.rnn_layers = rnn_layers
        self.fc_layers = fc_layers
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.random_seed = random_seed
        self.w2v_embeddings = embeddings
        self.num_classes = 0
        self.num_steps = 0
        self.softmax = None
        self.loss_op = None
        self.train_op = None
        self.accuracy = None

    def init_model(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_steps = len(max(X, key=len))
        self.logger.info('Building graph. Num steps: %s', self.num_steps)
        tf.set_random_seed(self.random_seed)
        super().init_model(X, y)

    @property
    def prediction(self):
        if self.softmax is None:
            self.x_t = tf.placeholder(tf.int64, shape=[None, self.num_steps],
                                       name='x_input')
            self.y_t = tf.placeholder(tf.int64, shape=[None], name='y_input')

            with tf.name_scope('embeddings'):
                np_emb = np.array(self.w2v_embeddings)
                embeddings = tf.get_variable(name="W", shape=np_emb.shape, trainable=False,
                                             initializer=tf.constant_initializer(np_emb))
                rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x_t)

            with tf.name_scope('dnn'):
                logits = self._rnn(rnn_inputs)
                softmax = tf.nn.softmax(logits, name='y_proba')

            self._logits = logits
            self.softmax = softmax
        return self.softmax

    @property
    def loss(self):
        if self.loss_op is None:
            with tf.name_scope('loss'):
                xent = tf.keras.losses.sparse_categorical_crossentropy(self.y_t,
                                                                self.softmax)
                self.loss_op = tf.reduce_mean(xent, name='loss')
        return self.loss_op

    @property
    def optimize(self):
        if self.train_op is None:
            with tf.name_scope('train'):
                optimizer = self.optimizer(self.learning_rate)
                self.train_op = optimizer.minimize(self.loss)
        return self.train_op

    @property
    def error(self):
        if self.accuracy is None:
            with tf.name_scope('accuracy'):
                y_pred = tf.argmax(self.prediction, axis=1, name='output')
                correct = tf.equal(y_pred, self.y_t)
                self.accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
        return self.accuracy

    def _assert_valid_params(self):
        if not isinstance(self.num_units, (list, tuple)):
            raise InvalidArgumentError('num_units',
                                       'Parameter num_units must be a list or tuple with the number of units per layer')
        elif len(self.num_units) == 0:
            raise InvalidArgumentError('num_units', 'Length of num units must be greater than zero')

    def _rnn(self, inputs):
        current_input = inputs
        self.is_training = tf.placeholder_with_default(False, shape=(),
                                                       name='is_training')
        kwargs = {'training': self.is_training}
        with tf.name_scope('rnn_layers'):
            for layer in self.rnn_layers:
                current_input = layer.build_tensor(current_input, **kwargs)
        with tf.name_scope('fc_layers'):
            for layer in self.fc_layers:
                current_input = layer.build_tensor(current_input, **kwargs)
            output = tf.keras.layers.Dense(self.num_classes)(current_input)
        return output
