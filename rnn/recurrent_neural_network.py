from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import tag_constants

from src.rnn.rnn_cells import GRUCellFactory
from src.exceptions import InvalidArgumentError
from src.rnn.utils import get_length_tensor, get_next_batch, \
                          he_init, last_relevant

import numpy as np
import tensorflow as tf

import logging
import os


class RecurrentNeuralNetwork(BaseEstimator, ClassifierMixin):
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
    >>> from src.recurrent_neural_network import RecurrentNeuralNetwork
    >>> X = ['Hi, how are you doing?']
    >>> X_prep = X
    """

    def __init__(self, num_epochs=30, batch_size=200, num_units=(256,),
                 cell_factory=GRUCellFactory(), activation=None,
                 kernel_initializer=he_init, layer_norm=False,
                 dropout_rate=0.0, learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer, save_dir='results/rnn',
                 logging_level=logging.WARNING, early_stopping=True,
                 max_epochs_without_progress=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_units = num_units
        self.cell_factory = cell_factory
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.logger.level = logging_level
        self.save_dir = save_dir
        self.save_path = os.path.join(self.save_dir, 'best.ckpt')
        self.early_stopping = early_stopping
        self.max_epochs_without_progress = max_epochs_without_progress \
            if max_epochs_without_progress is not None \
            else num_epochs

    def fit(self, x, y=None, **fit_params):
        embedding_size = len(x[0][0])
        num_classes = len(np.unique(y))
        max_length = len(max(x, key=len))

        tf.reset_default_graph()
        tf.set_random_seed(42)
        self._build_graph(max_length, embedding_size, num_classes)
        self.session = tf.Session()

        self.logger.info('Fitting %s...', str(self))
        self.logger.info('X shape: %s', np.shape(x))

        if self.early_stopping:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)
            self._train_loop(x_train, x_val, y_train, y_val)
        else:
            self._train_loop(x, None, y, None)

        self.logger.info('Restoring checkpoint of best model...')
        self.saver.restore(self.session, self.save_path)

    def _train_loop(self, x_train, x_val, y_train, y_val):
        best_loss = np.inf
        epochs_without_progress = 0

        num_batches = len(x_train) // self.batch_size
        self.session.run(self._init)
        for epoch in range(self.num_epochs):
            train_accuracy = 0
            for batch_idx in range(num_batches):
                x_batch, y_batch = get_next_batch(x_train, y_train,
                                                  batch_idx, self.batch_size)
                feed_dict = {
                    self._x_t: x_batch,
                    self._y_t: y_batch
                }
                _, batch_accuracy = self.session.run([self._train_op, self._accuracy], feed_dict=feed_dict)
                train_accuracy += batch_accuracy
            train_accuracy /= num_batches

            if x_val is not None:
                loss, accuracy = self.session.run([self._loss_op, self._accuracy], feed_dict={
                    self._x_t: x_val,
                    self._y_t: y_val
                })

                if loss < best_loss:
                    best_loss = loss
                    epochs_without_progress = 0
                    self.saver.save(self.session, self.save_path)
                else:
                    epochs_without_progress += 1
                    if epochs_without_progress > self.max_epochs_without_progress:
                        self.logger.info('No progress after %s epochs. '
                                         'Stopping...',
                                         self.max_epochs_without_progress)
                        break
                self.logger.info('Epoch {}\t - Loss: {:.7f} - Best loss: '
                                 '{:.7f} - Val Accuracy: {:.3f} - Train Accura'
                                 'cy: {:.3f}'.format(epoch, loss, best_loss,
                                                     accuracy * 100,
                                                     train_accuracy * 100))
            else:
                self.logger.info('Epoch: {}\t - Accuracy: {:.3f}'.format(
                                epoch, train_accuracy*100))
                self.saver.save(self.session, self.save_path)

    def save(self, save_path):
        inputs = {"x_t": self._x_t}
        outputs = {"pred_proba": self._y_proba}
        tf.saved_model.simple_save(self.session, save_path, inputs, outputs)

    def restore(self, save_path):
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)
        tf.saved_model.loader.load(
            self.session,
            [tag_constants.SERVING],
            save_path,
        )
        self._x_t = graph.get_tensor_by_name('x_input:0')
        self._y_proba = graph.get_tensor_by_name('dnn/y_proba:0')

    def predict(self, x):
        self.logger.info('Predict x: %s', np.shape(x))
        if self.session is None:
            raise NotFittedError
        else:
            if len(x) > 15000:
                x = np.asarray(x, dtype=np.float32)
            probabilities = self._predict_proba(x)
            return np.argmax(probabilities, axis=1)

    def _predict_proba(self, x):
        self.logger.info('Predict probabilities of x: %s', np.shape(x))
        with self.session.as_default():
            return self._y_proba.eval(feed_dict={
                self._x_t: x
            })

    def _build_graph(self, num_steps, embedding_size, num_classes):
        self.logger.info('Building graph. Num steps: %s, embedding size: %s', num_steps, embedding_size)

        x_t = tf.placeholder(tf.float32, shape=[None, num_steps, embedding_size], name='x_input')
        y_t = tf.placeholder(tf.int64, shape=[None], name='y_input')

        with tf.name_scope('dnn'):
            logits = self._rnn(x_t, num_classes)
            softmax = tf.nn.softmax(logits, name='y_proba')

        with tf.name_scope('loss'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_t, logits=logits, name='cross_entropy')
            loss_op = tf.reduce_mean(xent, name='loss')

        with tf.name_scope('train'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss_op)

        with tf.name_scope('accuracy'):
            y_pred = tf.argmax(softmax, axis=1, name='output')
            correct = tf.equal(y_pred, y_t)
            accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        self.logger.info('Graph built correctly')

        self._x_t = x_t
        self._y_t = y_t
        self._y_proba = softmax
        self._loss_op = loss_op
        self._train_op = train_op
        self._accuracy = accuracy
        self._init = init
        self.saver = saver

    def _assert_valid_params(self):
        if not isinstance(self.num_units, (list, tuple)):
            raise InvalidArgumentError('num_units',
                                       'Parameter num_units must be a list or tuple with the number of units per layer')
        elif len(self.num_units) == 0:
            raise InvalidArgumentError('num_units', 'Length of num units must be greater than zero')

    def _rnn(self, inputs, num_classes):
        # TODO: check if we need to pass is_training tensor to cell factory
        if len(self.num_units) > 1:
            # multiple layers
            cells = [self.cell_factory(units, self.activation, self.kernel_initializer,
                                       self.dropout_rate, self.layer_norm)
                     for units in self.num_units]
            final_cell = tf.contrib.rnn.MultiRNNCell(cells)
        else:
            # single layer
            final_cell = self.cell_factory(self.num_units[0], self.activation, self.kernel_initializer,
                                           self.dropout_rate, self.layer_norm)
        rnn_output, _ = tf.nn.dynamic_rnn(final_cell, inputs,
                                          sequence_length=get_length_tensor(inputs),
                                          dtype=tf.float32)
        last = last_relevant(rnn_output, get_length_tensor(inputs))

        with tf.name_scope('fc_layer'):
            output = tf.layers.dense(last, num_classes, kernel_initializer=self.kernel_initializer)
        return output

    def __repr__(self):
        return "rnn_bs{}_nc{}_ce{}_ln{}_dr{}_lr{}_op{}".format(
            self.batch_size, self.num_units, self.cell_factory.__class__.__name__,
            self.layer_norm, self.dropout_rate, self.learning_rate,
            self.optimizer.__name__)

    def __str__(self):
        return "RNN. batch_size: {} - num_cells: {} - cell: {} \
            - norm: {} - dropout: {} - l_rate: {} - optimizer: {}".format(
            self.batch_size, self.num_units, self.cell_factory.__class__.__name__,
            self.layer_norm, self.dropout_rate, self.learning_rate,
            self.optimizer.__name__)
