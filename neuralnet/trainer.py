import logging
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from .utils import get_next_batch


class BaseTrainer(ABC):
    def __init__(self, model, batch_size=30, num_epochs=200):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.X_train = []
        self.y_train = []
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def train(self, X, y):
        self.logger.info('Creating model...')
        self.model.init_model(X, y)

        self.logger.info('Starting training...')
        self._train_loop(X, y)
        self.model.on_train_finished()

    def _train_loop(self, X, y):
        self.on_train_loop_started(X, y)
        num_batches = len(self.X_train) // self.batch_size
        self.model.session.run(self._init)
        for epoch in range(self.num_epochs):
            self._epoch_loop(epoch, num_batches)

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
                self.model.keep_prob: 1 - self.model.dropout_rate
            }
            batch_loss, batch_accuracy = self.model.session.run(
                [self.model._train_op, self.model._accuracy], feed_dict=feed_dict)
            train_accuracy += batch_accuracy
            train_loss += batch_loss
        train_accuracy /= num_batches
        train_loss /= num_batches
        self.on_epoch_finished(train_loss, train_accuracy)

    @abstractmethod
    def on_train_loop_started(self, X, y):
        self.X_train = X
        self.y_train = y

    @abstractmethod
    def on_epoch_finished(self, loss, acc):
        self.logger.info('Train: Loss = {:.3f} - Accuracy = {:.3f}'.format(
            loss, acc * 100))
        self.model.on_epoch_finished(loss, acc)


class EarlyStoppingTrainer(BaseTrainer):
    def __init__(self, model, batch_size, num_epochs,
                 validation_split=0.2, max_epochs_no_progress=5,
                 random_seed=42):
        super(BaseTrainer, self).__init__(model, batch_size, num_epochs)
        self.validation_split = validation_split
        self.max_epochs_without_progress = max_epochs_no_progress
        self.random_seed = random_seed
        self.best_loss = np.inf
        self.epochs_without_progress = 0
        self.X_val = []
        self.y_val = []

    def on_train_loop_started(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X, y, test_size=self.validation_split,
                             random_state=self.random_seed)

    def on_epoch_finished(self, loss, acc):
        val_loss, val_acc = self.model.session.run(
            [self._loss_op, self._accuracy],
            feed_dict={self._x_t: self.X_val, self._y_t: self.y_val})

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_progress = 0
        else:
            self.epochs_without_progress += 1
            if self.epochs_without_progress > self.max_epochs_without_progress:
                self.logger.info('No progress after %s epochs. '
                                 'Stopping...',
                                 self.max_epochs_without_progress)
                return
        self.logger.info('Validation: Loss = {:.5f} - Best loss = '
                         '{:.5f} - Accuracy = {:.3f}'.format(val_loss,
                                                             self.best_loss,
                                                             val_acc * 100))
        super().on_epoch_finished(loss, acc)
