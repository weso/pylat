import logging
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from pylat.neuralnet.rnn.recurrent_neural_network import RecurrentNeuralNetwork
from pylat.neuralnet.trainer import BaseTrainer, EarlyStoppingTrainer


class RNNWrapper(BaseEstimator, ClassifierMixin):

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

    def fit(self, X, y, **fit_args):
        self.trainer.train(X, y)

    def predict(self, x, **params):
        self.logger.info('Predict x: %s', np.shape(x))
        if self.model.session is None:
            raise NotFittedError
        else:
            probabilities = self._predict_proba(x, **params)
            return np.argmax(probabilities, axis=1)

    def _predict_proba(self, x, **params):
        self.logger.info('Predict probabilities of x: %s', np.shape(x))
        with self.model.session.as_default():
            self.model.additional_inits(**params)
            return self.model.prediction.eval(feed_dict={
                self.model.x_t: x
            })
