import logging
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from pylat.neuralnet.rnn.recurrent_neural_network import RecurrentNeuralNetwork
from pylat.neuralnet.trainer import BaseTrainer, EarlyStoppingTrainer


class RNNWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper around the recurrent neural network that fits the sklearn API.

    This class is meant to create and train easily a recurrent neural network
    that can be used in a scikit learn pipeline.

    Parameters
    ----------
    rnn_layers : :obj:`list` of :obj:`RecurrentLayer`
        List of recurrent layers that will compose the neural network. These
        layers will be the first ones to receive the input of the embeddings.
    fc_layers : :obj:`list` of :obj:`DenseLayer`
        List of dense layers that will receive the output of the recurrent
        layers.
    num_epochs : int, optional (default=200)
        Maximum number of epochs taken during the training phase.
    batch_size : int, optional (default=30)
        Number of training instances used for each gradient update
        in the training phase.
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
    early_stopping : bool
        Whether early stopping will be applied in the training phase or not.
    validation_split : float, optional (default=0.2)
        If early stopping is set to True, percentage of the training data that
        should be separated to validation purposes.
    validation_data : :obj:`tuple`, optional (default=None)
        If early stopping is set to True, tuple of the form (X_val, y_val) with
        the validation instances and labels used in the training phase. If this
        tuple is provided, the validation_split parameter will be ignored.
    max_epochs_no_progress : int, optional (default=5)
        If early stopping is set to True, maximum number of epochs that the
        network will be trained with consecutive increases in the loss. After
        this number is surpassed the training will be stopped.
    """

    def __init__(self, rnn_layers, fc_layers, num_epochs=30,
                 batch_size=200, learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer, save_dir='results',
                 random_seed=42, embeddings=None, early_stopping=False,
                 validation_split=0.2, validation_data=None,
                 max_epochs_no_progress=5):
        self.model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                            learning_rate, optimizer,
                                            save_dir, random_seed, embeddings)
        self.trainer = BaseTrainer(self.model, num_epochs, batch_size) \
            if not early_stopping \
            else EarlyStoppingTrainer(self.model, num_epochs, batch_size,
                                      validation_split, validation_data,
                                      max_epochs_no_progress)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def fit(self, X, y):
        """Train the neural network with the given data

        Parameters
        ----------
        X : :obj:`list` of :obj:`list` of int
            A 2 dimensional list, where the first dimension contains every
            sentence used to train the embedding, and the second dimension
            contains the id of every token of each sentence.
        y : :obj:`list`
            Labels of the passed data.
        """
        self.trainer.train(X, y)

    def predict(self, x):
        """Predict the label of the given instances.

        Parameters
        ----------
        x : :obj:`list` of :obj:`list` of int
            A 2 dimensional list, where the first dimension contains every
            sentence used to train the embedding, and the second dimension
            contains the id of every token of each sentence.

        Returns
        -------
        :obj:`list` of int
            Predicted labels of each instance.
        """
        self.logger.info('Predict x: %s', np.shape(x))
        if self.model.session is None:
            raise NotFittedError
        else:
            probabilities = self.predict_proba(x)
            return np.argmax(probabilities, axis=1)

    def predict_proba(self, x):
        """Predict the probabilities of the data belonging to each label.

        Parameters
        ----------
        x : :obj:`list` of :obj:`list` of int
            A 2 dimensional list, where the first dimension contains every
            sentence used to train the embedding, and the second dimension
            contains the id of every token of each sentence.

        Returns
        -------
        :obj:`list` of :obj:`list` of float
            A 2 dimensional list, where the first dimension corresponds to
            every instance passed as input and the second dimension contains
            the probability values between 0 and 1 of the instance belonging
            to each label.
        """
        self.logger.info('Predict probabilities of x: %s', np.shape(x))
        with self.model.session.as_default():
            self.model.additional_inits()
            return self.model.prediction.eval(feed_dict={
                self.model.x_t: x
            })

    def save(self, save_path):
        """Saves the neural network to the given path.

        Parameters
        ----------
        save_path : str
            Directory where the neural network will be saved.
        """
        self.model.save(save_path)

    def restore(self, save_path):
        """Restores the neural network from the given path.

        The tensors and their weights will be recovered from the saved files
        in the directory.

        Parameters
        ----------
        save_path : str
            Directory that holds the saved data of the neural network to be
            restored.
        """
        self.model.restore(save_path)
