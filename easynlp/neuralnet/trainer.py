import logging
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from .utils import get_next_batch


class BaseTrainer(ABC):
    """

    Parameters
    ----------
    num_epochs: int (default=200)
        Maximum number of epochs taken during the training phase.

    batch_size: int (default=30)
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

    def train(self, X, y):
        self.logger.info('Creating model...')
        self.model.init_model(X, y)
        self.saver = tf.train.Saver()

        self.logger.info('Starting training...')
        self._train_loop(X, y)
        self.on_train_finished()

    def _train_loop(self, X, y):
        self.on_train_loop_started(X, y)
        num_batches = len(self.X_train) // self.batch_size + 1
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
        self.X_train = X
        self.y_train = y

    def on_epoch_finished(self, loss, acc):
        self.logger.info('Train: Loss = {:.3f} - Accuracy = {:.3f}'.format(
            loss, acc * 100))
        return False

    def on_train_finished(self):
        pass


class EarlyStoppingTrainer(BaseTrainer):
    def __init__(self, model, batch_size, num_epochs,
                 validation_split=0.2, validation_data=None,
                 max_epochs_no_progress=5, random_seed=42):
        super().__init__(model, batch_size, num_epochs)
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
