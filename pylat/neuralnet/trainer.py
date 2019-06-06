import logging
import math
import numpy as np
import tensorflow as tf

from abc import ABC
from sklearn.model_selection import train_test_split
from .utils import get_next_batch


class BaseTrainer(ABC):
    """Base class of neural network trainers.

    This class provides a basic implementation for training any type of neural
    network that conforms to the interface provided by the
    :obj:`BaseNeuralNetwork` class.

    Parameters
    ----------
    model : :obj:`BaseNeuralNetwork` subclass
        Neural network model to be trained.
    num_epochs : int, optional (default=200)
        Maximum number of epochs taken during the training phase.
    batch_size : int, optional (default=30)
        Number of training instances used for each gradient update
        in the training phase.
    """

    def __init__(self, model, num_epochs=200, batch_size=30):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.X_train = []
        self.y_train = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.saver = None
        self._epochs_completed = 0

    def train(self, X, y):
        """Train the neural network model with the given data.

        Parameters
        ----------
        X : :obj:`np.array`
            Two dimensional array containing the training instances.
        y : list
            List containing the labels of each instance.
        """
        self.logger.info('Creating model...')
        self.model.init_model(X, y)
        self.saver = tf.train.Saver()

        self.logger.info('Starting training...')
        self._train_loop(X, y)
        self.on_train_finished()

    def _train_loop(self, X, y):
        self.on_train_loop_started(X, y)
        num_batches = math.ceil(len(self.X_train) / self.batch_size)
        for epoch in range(self.num_epochs):
            stop_train = self._epoch_loop(epoch, num_batches)
            if stop_train:
                break

    def _epoch_loop(self, num_epoch, num_batches):
        self.logger.info('Epoch: {}'.format(num_epoch))
        train_accuracy = 0
        train_loss = 0
        for batch_idx in range(num_batches):
            x_batch, y_batch = get_next_batch(self.X_train, self.y_train,
                                              batch_idx, self.batch_size)
            feed_dict = {
                self.model.x_t: x_batch,
                self.model.y_t: y_batch,
                self.model.is_training: True
            }
            _, batch_loss, batch_accuracy = self.model.session.run(
                [self.model.optimize, self.model.loss, self.model.error],
                feed_dict=feed_dict)
            train_accuracy += batch_accuracy
            train_loss += batch_loss
        train_accuracy /= num_batches
        train_loss /= num_batches
        return self.on_epoch_finished(train_loss, train_accuracy)

    def on_train_loop_started(self, X, y):
        """Perform additional operations before the beginning of the first epoch

        Parameters
        ----------
        X : :obj:`np.array`
            Two dimensional array containing the training instances.
        y : list
            List containing the labels of each instance.
        """
        self._epochs_completed = 0
        self.X_train = X
        self.y_train = y

    def on_epoch_finished(self, loss, acc):
        """Perform additional operations once a training epoch is finished.

        Parameters
        ----------
        loss : float
            Loss value between 0 and 1 of the network at the end of the epoch.
        acc : float
            Accuracy value between 0 and 1of the network at the end of the
            epoch.

        Returns
        -------
        bool
            Whether the training should stop after this epoch.
        """
        self.logger.info('Train: Loss = {:.3f} - Accuracy = {:.3f}'.format(
            loss, acc * 100))
        self._epochs_completed += 1
        return False

    def on_train_finished(self):
        """Perform additional operations once the training is finished.

        This method can be overridden by other training classes to
        perform additional steps once all of the training epochs have
        been finished.
        """
        pass


class EarlyStoppingTrainer(BaseTrainer):
    """Train a neural network with early stopping.

    This class inherits from the :obj:`BaseTrainer` class to provide early
    stopping capabilities to the training phase. This means that a validation
    set will be used to evaluate how the network is performing after each
    training epoch. If the loss is increasing several epochs in a row, the
    training will finish and the weights of the network where the loss was
    minimized will be restored.

    Parameters
    ----------
    model : :obj:`BaseNeuralNetwork` subclass
        Neural network model to be trained.
    num_epochs : int
        Maximum number of epochs taken during the training phase.
    batch_size : int
        Number of training instances used for each gradient update
        in the training phase.
    validation_split : float, optional (default=0.2)
        Percentage of the training data that should be separated to
        validation purposes.
    validation_data : :obj:`tuple`, optional (default=None)
        Tuple of the form (X_val, y_val) with the validation instances and
        labels used in the training phase. If this tuple is provided, the
        validation_split parameter will be ignored.
    max_epochs_no_progress : int, optional (default=5)
        Maximum number of epochs that the network will be trained with
        consecutive increases in the loss. After this number is surpassed
        the training will be stopped.
    random_seed : int, optional (default=42)
        Random seed used to divide the training data into a validation and
        train sets.
    """

    def __init__(self, model, num_epochs, batch_size,
                 validation_split=0.2, validation_data=None,
                 max_epochs_no_progress=5, random_seed=42):
        super().__init__(model, num_epochs, batch_size)
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.max_epochs_without_progress = max_epochs_no_progress
        self.random_seed = random_seed
        self.best_loss = np.inf
        self.epochs_without_progress = 0
        self.X_val = []
        self.y_val = []

    def on_train_loop_started(self, X, y):
        if self.validation_data is None:
            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(X, y, test_size=self.validation_split,
                                 random_state=self.random_seed)
        else:
            self.X_val = self.validation_data[0]
            self.y_val = self.validation_data[1]
            self.X_train = X
            self.y_train = y

    def on_epoch_finished(self, loss, acc):
        val_loss, val_acc = self.model.session.run(
            [self.model.loss, self.model.error],
            feed_dict={self.model.x_t: self.X_val, self.model.y_t: self.y_val})

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_progress = 0
            self.saver.save(self.model.session, self.model.save_path)
        else:
            self.epochs_without_progress += 1
            if self.epochs_without_progress > self.max_epochs_without_progress:
                self.logger.info('No progress after %s epochs. '
                                 'Stopping...',
                                 self.max_epochs_without_progress)
                return True
        self.logger.info('Validation: Loss = {:.5f} - Best loss = '
                         '{:.5f} - Accuracy = {:.3f}'.format(val_loss,
                                                             self.best_loss,
                                                             val_acc * 100))
        return super().on_epoch_finished(loss, acc)

    def on_train_finished(self):
        self.saver.restore(self.model.session, self.model.save_path)
