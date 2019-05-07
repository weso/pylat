from pylat.exceptions import InvalidArgumentError
from pylat.neuralnet.embeddings import Word2VecEmbedding
from pylat.neuralnet.layers import DenseLayer
from pylat.neuralnet.trainer import BaseTrainer, EarlyStoppingTrainer
from pylat.neuralnet.rnn.cells import GRUCellFactory, LSTMCellFactory, \
    SimpleCellFactory
from pylat.neuralnet.rnn.layers import BidirectionalRecurrentLayer, RecurrentLayer
from pylat.neuralnet.rnn.recurrent_neural_network import RecurrentNeuralNetwork

from gensim.models import Word2Vec

import tensorflow as tf
import os
import pytest
import unittest


class TestRecurrentNeuralNetwork(unittest.TestCase):

    def setUp(self):
        embeddings_dir = os.path.join('test', 'data', 'embeddings')
        model = Word2Vec.load(os.path.join(embeddings_dir,
                                           'w2v_test.model'))
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

    def test_base_training(self):
        rnn_layers = [RecurrentLayer(num_units=3, cell_factory=GRUCellFactory())]
        fc_layers = [DenseLayer(num_units=5)]
        model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                       embeddings=self.embeddings)
        trainer = BaseTrainer(model, num_epochs=100, batch_size=1)
        trainer.train(self.x_train, self.y_train)
        self._assert_op_in_graph(model, 'Adam', 'dnn/rnn_layers')
        self._assert_op_in_graph(model, 'rnn_layer', 'dnn/rnn_layers')
        self._assert_op_in_graph(model, 'Adam', 'dnn/fc_layers')
        self._assert_op_in_graph(model, 'dense_layer', 'dnn/fc_layers')
        assert trainer._epochs_completed == 100

    def test_early_stopping(self):
        rnn_layers = [RecurrentLayer(num_units=3,
                                     cell_factory=SimpleCellFactory())]
        fc_layers = [DenseLayer(num_units=5)]
        model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                       embeddings=self.embeddings)
        trainer = EarlyStoppingTrainer(model, num_epochs=100, batch_size=1,
                                       max_epochs_no_progress=5,
                                       validation_data=(self.x_val, self.y_val))
        trainer.train(self.x_train, self.y_train)
        assert trainer._epochs_completed < 100  # early stopping

    def test_dropout_layer_norm(self):
        rnn_layers = [BidirectionalRecurrentLayer(num_units=2,
                                                  dropout_rate=0.34,
                                                  layer_norm=True,
                                                  cell_factory=LSTMCellFactory()
                                                  )]
        fc_layers = [DenseLayer(num_units=15), DenseLayer(num_units=5)]
        model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                       embeddings=self.embeddings)
        trainer = EarlyStoppingTrainer(model, num_epochs=3, batch_size=1,
                                       max_epochs_no_progress=2)
        trainer.train(self.x_train, self.y_train)
        assert trainer._epochs_completed == 3  # all epochs completed
        self._assert_op_in_graph(model, 'bidi_rnn', 'dnn/rnn_layers')
        self._assert_op_in_graph(model, 'forward_rnn_layer', 'dnn/rnn_layers')
        self._assert_op_in_graph(model, 'backward_rnn_layer', 'dnn/rnn_layers')
        self._assert_op_in_graph(model, 'dense_layer', 'dnn/fc_layers')
        self._assert_op_in_graph(model, 'dense_layer_1', 'dnn/fc_layers')

    def test_dropout_invalid(self):
        with pytest.raises(InvalidArgumentError):
            rnn_layers = [RecurrentLayer(num_units=5, dropout_rate=-0.34)]
            fc_layers = [DenseLayer(num_units=15)]
            model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                           embeddings=self.embeddings)
            trainer = BaseTrainer(model)
            trainer.train(self.x_train, self.y_train)

    def test_invalid_data(self):
        with pytest.raises(InvalidArgumentError):
            rnn_layers = [RecurrentLayer(num_units=5)]
            fc_layers = [DenseLayer(num_units=15)]
            model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                           embeddings=self.embeddings)
            trainer = BaseTrainer(model)
            trainer.train([1, 2, 3, 4], [1, 2, 3, 4])

    def test_invalid_layers(self):
        with pytest.raises(InvalidArgumentError):
            model = RecurrentNeuralNetwork(rnn_layers=[], fc_layers=[],
                                           embeddings=self.embeddings)
            trainer = BaseTrainer(model)
            trainer.train(self.x_train, self.y_train)

    def test_invalid_cell(self):
        with pytest.raises(InvalidArgumentError):
            rnn_layers = [RecurrentLayer(num_units=5, cell_factory=self.x_train)]
            fc_layers = [DenseLayer(num_units=15)]
            model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                           embeddings=self.embeddings)
            trainer = BaseTrainer(model)
            trainer.train(self.x_train, self.y_train)

    def _assert_op_in_graph(self, model, op_name, scope,
                            op_type=tf.GraphKeys.GLOBAL_VARIABLES):
        for i in model.session.graph.get_collection(op_type, scope):
            if op_name in i.name:
                return
        self.assertFalse(True, 'Operation {} was not found '
                               'in the model'.format(op_name))
