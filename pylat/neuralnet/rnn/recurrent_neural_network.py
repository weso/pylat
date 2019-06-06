from ..model import BaseNeuralNetwork
from ...exceptions import InvalidArgumentError

from tensorflow.python.saved_model import tag_constants

import numpy as np
import tensorflow as tf

import logging


class RecurrentNeuralNetwork(BaseNeuralNetwork):
    """Recurrent neural network implementation using TensorFlow.

    This class inherits from the :obj:`BaseNeuralNetwork` abstract class and
    provides an example of a recurrent neural network implementation using
    TensorFlow.

    It can be fully customizable using different types of layers, and can be
    trained with any class that fulfills the :obj:`BaseTrainer` interface.

    Parameters
    ----------
    rnn_layers : :obj:`list` of :obj:`RecurrentLayer`
        List of recurrent layers that will compose the neural network. These
        layers will be the first ones to receive the input of the embeddings.
    fc_layers : :obj:`list` of :obj:`DenseLayer`
        List of dense layers that will receive the output of the recurrent
        layers.
    learning_rate : float, optional (default=1e-4)
        Learning rate used by the optimizer during the training phase.
    optimizer : :obj:`tf.train.Optimizer` class (default=tf.train.AdamOptimizer)
        Optimizer used during the training phase to minimize the loss
        function.
    save_dir : str, optional (default='results')
        Directory where the neural network will be saved during training.
    random_seed : int, optional (default=42)
        Random seed used to initialize the TensorFlow operations.
    embeddings : :obj:`WordEmbeddings`, optional (default=None)
        WordEmbeddings class used to load the word vectors from the input data.

    Attributes
    ----------
    session : tf.Session
        TensorFlow session where the neural network operations will
        be executed.

    Examples
    --------
    >>> from pylat.neuralnet.rnn.recurrent_neural_network import \
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
        self.random_seed = random_seed
        self.w2v_embeddings = embeddings
        self.softmax = None
        self.loss_op = None
        self.train_op = None
        self.accuracy = None
        self._num_classes = 0
        self._num_steps = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._check_valid_params()

    def init_model(self, X, y):
        RecurrentNeuralNetwork._assert_valid_input(X, y)
        self._check_input(X, y)
        self.logger.info('Building graph. Num steps: %s', self._num_steps)
        tf.set_random_seed(self.random_seed)
        super().init_model(X, y)

    @property
    def prediction(self):
        if self.softmax is None:
            with tf.name_scope('input'):
                rnn_inputs = self._input()

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

    def _check_input(self, X, y):
        self._num_classes = len(np.unique(y))
        self._num_steps = len(max(X, key=len))

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
            output = tf.keras.layers.Dense(self._num_classes)(current_input)
        return output

    def _input(self):
        self.x_t = tf.placeholder(tf.int64, shape=[None, self._num_steps],
                                  name='x_input')
        self.y_t = tf.placeholder(tf.int64, shape=[None], name='y_input')
        with tf.name_scope('embeddings'):
            np_emb = np.array(self.w2v_embeddings.get_vectors())
            vocab_size, emb_dim = np.shape(np_emb)
            embeddings = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, emb_dim]),
                trainable=False, name="W")
            self._emb_placeholder = tf.placeholder(tf.float32,
                                                   [vocab_size, emb_dim],
                                                   name='emb_plh')
            self._emb_init = embeddings.assign(self._emb_placeholder,
                                               name='emb_init')
            rnn_inputs = tf.nn.embedding_lookup(embeddings, self.x_t)
        return rnn_inputs

    def additional_inits(self):
        self.session.run(self._emb_init, feed_dict={
            self._emb_placeholder: self.w2v_embeddings.get_vectors()
        })

    def save(self, save_path):
        inputs = {"x_t": self.x_t}
        outputs = {"pred_proba": self.prediction}
        tf.saved_model.simple_save(self.session, save_path, inputs, outputs)

    def restore(self, save_path):
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)
        tf.saved_model.loader.load(
            self.session,
            [tag_constants.SERVING],
            save_path,
        )
        self.x_t = graph.get_tensor_by_name('input/x_input:0')
        self.softmax = graph.get_tensor_by_name('dnn/y_proba:0')
        self._emb_init = graph.get_tensor_by_name('input/embeddings/emb_init:0')
        self._emb_placeholder = graph.get_tensor_by_name('input/embeddings/'
                                                         'emb_plh:0')

    def _check_valid_params(self):
        if len(self.fc_layers) == 0 or len(self.rnn_layers) == 0:
            raise InvalidArgumentError('layers', 'Layers must not be empty.')

    @staticmethod
    def _assert_valid_input(X, y):
        try:
            np_x = np.asarray(X)
            np_y = np.asarray(y)
        except Exception:
            raise InvalidArgumentError('x/y', 'X and y must be iterables.')

        if len(np_x.shape) != 2:
            raise InvalidArgumentError('x',
                                       'X array must have 2 dimensions.')
        if len(np_y.shape) != 1:
            raise InvalidArgumentError('y',
                                       'y array must have 1 dimension.')
