from gensim.models import Word2Vec

from pylat.neuralnet import BaseTrainer, DenseLayer, EarlyStoppingTrainer, \
    Word2VecEmbedding
from pylat.neuralnet.rnn import GRUCellFactory, RecurrentLayer
from pylat.wrapper.predictor import RNNWrapper
from sklearn.exceptions import NotFittedError

import numpy as np
import os
import pytest
import shutil
import tensorflow as tf
import unittest


class TestRNNWrapper(unittest.TestCase):
    def setUp(self):
        self.rnn_layers = [RecurrentLayer(num_units=3,
                                          cell_factory=GRUCellFactory())]
        self.fc_layers = [DenseLayer(num_units=5)]
        embeddings_dir = os.path.join('test', 'data', 'embeddings')
        model = Word2Vec.load(os.path.join(embeddings_dir, 'w2v_test.model'))
        self.embeddings = Word2VecEmbedding(model=model)
        self.x_train = [[0, 2, 1, 0],
                        [2, 1, 0, 0],
                        [3, 1, 0, 1],
                        [2, 0, 0, 0],
                        [1, 3, 3, 1]]
        self.y_train = [0, 0, 1, 0, 1]
        self.x_val = [
            [0, 1, 2, 0],
            [2, 1, 2, 3]
        ]
        self.y_val = [0, 1]
        tf.reset_default_graph()

    def test_early_stopping(self):
        model = RNNWrapper(self.rnn_layers, self.fc_layers,
                           embeddings=self.embeddings, early_stopping=True)
        assert isinstance(model.trainer, EarlyStoppingTrainer)

    def test_base_training(self):
        model = RNNWrapper(self.rnn_layers, self.fc_layers,
                           embeddings=self.embeddings)
        assert isinstance(model.trainer, BaseTrainer)

    def test_not_fitted(self):
        model = RNNWrapper(self.rnn_layers, self.fc_layers,
                           embeddings=self.embeddings)
        with pytest.raises(NotFittedError):
            model.predict(self.x_val)

    def test_save_restore(self):
        model = RNNWrapper(self.rnn_layers, self.fc_layers,
                           embeddings=self.embeddings)
        model.fit(self.x_train, self.y_train)
        old_prediction = model.predict(self.x_val)
        model.save('tmp')
        new_model = RNNWrapper(self.rnn_layers, self.fc_layers,
                               embeddings=self.embeddings)
        new_model.restore('tmp')
        assert np.allclose(old_prediction, new_model.predict(self.x_val))
        shutil.rmtree('tmp')


if __name__ == '__main__':
    unittest.main()
