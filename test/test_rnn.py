from pylat.exceptions import InvalidArgumentError
from pylat.neuralnet import BaseTrainer, DenseLayer, EarlyStoppingTrainer, \
    LayerConfig, Word2VecEmbedding
from pylat.neuralnet.rnn import GRUCellFactory, LSTMCellFactory, \
    SimpleCellFactory, BidirectionalRecurrentLayer, RecurrentLayer,\
    RecurrentNeuralNetwork
from pylat.wrapper.predictor import RNNWrapper

from gensim.models import Word2Vec

import numpy as np
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
        fc_layers = [DenseLayer(num_units=15, dropout_rate=0.3),
                     DenseLayer(num_units=5, dropout_rate=0.1)]
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
        rnn_layers = [RecurrentLayer(num_units=5)]
        fc_layers = [DenseLayer(num_units=15)]
        model = RecurrentNeuralNetwork(rnn_layers, fc_layers,
                                       embeddings=self.embeddings)
        trainer = BaseTrainer(model)
        with pytest.raises(InvalidArgumentError):
            trainer.train([1, 2, 3, 4], [1, 2, 3, 4])  # x must be 2d
        with pytest.raises(InvalidArgumentError):
            trainer.train([[1, 2], [2, 3]], 0)  # y must be 1d
        with pytest.raises(InvalidArgumentError):
            trainer.train('asd', 'fgh')  # data must be numbers

    def test_invalid_layers(self):
        with pytest.raises(InvalidArgumentError):
            model = RecurrentNeuralNetwork(rnn_layers=[], fc_layers=[],
                                           embeddings=self.embeddings)
            trainer = BaseTrainer(model)
            trainer.train(self.x_train, self.y_train)

    def test_layer_config_valid(self):
        config = LayerConfig(num_units=5)
        rnn_layer = RecurrentLayer(num_units=2)
        assert rnn_layer.num_units == 2
        rnn_layer.load_config(config)
        assert rnn_layer.num_units == 5

    def test_layer_config_invalid(self):
        with pytest.raises(InvalidArgumentError):
            config = 5
            rnn_layer = RecurrentLayer(num_units=2)
            rnn_layer.load_config(config)

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


class TestMetamorphicRelations(unittest.TestCase):
    def setUp(self):
        self._load_embeddings()
        self._load_data()
        self._load_model()

    def test_mr_additional_samples(self):
        """Test the additional samples metamorphic relationship.

        If additional samples with have label li are added before training the
        model, previous samples that were predicted with label li should still
        have the same label.
        """
        self.x_train.append([2, 1, 0, 0])
        self.y_train.append(0)
        self.model.fit(self.x_train, self.y_train)
        assert self.model.predict(self.x_val)[0] == 0

    def test_mr_permutation_attributes(self):
        """Test the attributes permutation metamorphic relationship.

        If the attributes are permutated, output should still be the same.
        """
        self.model.fit(self.x_train, self.y_train)
        output = self.model.predict(self.x_val)
        x_train_perm = [[3, 0, 0, 0],
                        [3, 0, 0, 0],
                        [3, 1, 1, 1],
                        [3, 0, 0, 0],
                        [3, 2, 2, 2],
                        [3, 1, 1, 1]]
        self._load_model()
        self.model.fit(x_train_perm, self.y_train)
        assert np.array_equal(output, self.model.predict(self.x_val))

    def test_mr_prediction_consistence(self):
        """Test the prediction consistence metamorphic relationship.

        Output should be the same in every prediction on the same data.
        """
        self.model.fit(self.x_train, self.y_train)
        output = self.model.predict(self.x_val)
        for i in range(50):
            assert np.array_equal(output, self.model.predict(self.x_val))

    def test_mr_removal_samples(self):
        """Test the samples removal metamorphic relationship.

        If some samples whose label is not li are removed, output
        for data which is predicted as li should still be the same.
        """
        del self.x_train[3]
        del self.y_train[3]
        self.model.fit(self.x_train, self.y_train)
        assert self.model.predict(self.x_val)[0] == 0

    def test_mr_removal_classes(self):
        """Test the classes removal metamorphic relationship.

        If some class which is not is not li is removed, output
        for data which is predicted as li should still be the same.
        """
        del self.x_train[3]
        del self.y_train[3]
        del self.x_train[4]
        del self.y_train[4]
        self.model.fit(self.x_train, self.y_train)
        assert self.model.predict(self.x_val)[0] == 0

    def _load_embeddings(self):
        embeddings_dir = os.path.join('test', 'data', 'embeddings')
        model = Word2Vec.load(os.path.join(embeddings_dir,
                                           'w2v_test.model'))
        self.embeddings = Word2VecEmbedding(model=model)

    def _load_data(self):
        self.x_train = [[0, 3, 0, 0],
                        [0, 3, 0, 0],
                        [1, 3, 1, 1],
                        [0, 3, 0, 0],
                        [2, 3, 2, 2],
                        [1, 3, 1, 1]]
        self.y_train = [0, 0, 1, 0, 2, 1]
        self.x_val = [
            [0, 3, 0, 0],
            [1, 3, 1, 1],
            [2, 3, 2, 2]
        ]
        self.y_val = [0, 1, 2]

    def _load_model(self):
        tf.reset_default_graph()
        rnn_layers = [RecurrentLayer(num_units=3,
                                     cell_factory=GRUCellFactory())]
        fc_layers = [DenseLayer(num_units=5)]
        self.model = RNNWrapper(rnn_layers, fc_layers,
                                embeddings=self.embeddings, num_epochs=6)
