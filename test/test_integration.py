from pylat.neuralnet import CrossLingualPretrainedEmbedding, DenseLayer, \
    Doc2VecEmbedding
from pylat.neuralnet.rnn import GRUCellFactory, RecurrentLayer
from pylat.wrapper.predictor import RNNWrapper
from pylat.wrapper.transformer import DocumentEmbeddingsTransformer, \
    SentencePadder, TextPreprocessor, WordEmbeddingsTransformer

from gensim.models import Doc2Vec
from sklearn.pipeline import Pipeline

import numpy as np
import os
import pytest
import unittest


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.valid_input = [
            "This is my friend, he is angry.",
            "Hello, why are you angry?"
        ]
        self.invalid_input = [
            ["this", "is", "already", "tokenized"]
        ]
        self.embeddings_dir = os.path.join('test', 'data', 'embeddings')
        self.preprocessor = TextPreprocessor(remove_stop_words=True)
        self.preprocessor.fit([])
        doc2vec_model = Doc2Vec.load(os.path.join(self.embeddings_dir,
                                                  'd2v_test.model'))
        self.doc_emb = DocumentEmbeddingsTransformer(Doc2VecEmbedding(
            model=doc2vec_model))
        self.word_emb = WordEmbeddingsTransformer(
            CrossLingualPretrainedEmbedding(self.embeddings_dir, 'en')
        )
        self.padder = SentencePadder(padding_length=5)
        self.rnn_layers = [RecurrentLayer(num_units=3,
                                          cell_factory=GRUCellFactory())]
        self.fc_layers = [DenseLayer(num_units=5)]

    def test_add_elements(self):
        pipe = Pipeline([('prepro', self.preprocessor)])
        assert np.array_equal(pipe.transform(self.valid_input),
                              [['this', 'friend', 'angry'], ['hello', 'angry']])
        pipe.steps.append(('vectors', self.doc_emb))
        assert np.shape(pipe.transform(self.valid_input)) == (2, 5)

    def test_base_pipe(self):
        pipe = Pipeline([('prepro', self.preprocessor),
                         ('vectors', self.doc_emb)])
        assert np.shape(pipe.transform(self.valid_input)) == (2, 5)

    def test_embeddings_pipe(self):
        pipe = Pipeline([('prepro', self.preprocessor),
                         ('vectors', self.word_emb),
                         ('padder', self.padder)])
        expected = [
            [2, 0, 0, 0, 0],
            [3, 0, 0, 0, 0]
        ]
        assert np.array_equal(pipe.transform(self.valid_input), expected)

    def test_invalid_input(self):
        pipe = Pipeline([('prepro', self.preprocessor),
                         ('vectors', self.doc_emb)])
        with pytest.raises(TypeError):
            pipe.transform(self.invalid_input)

    def test_predictor(self):
        rnn = RNNWrapper(self.rnn_layers, self.fc_layers)
        pipe = Pipeline([('prepro', self.preprocessor),
                         ('vectors', self.word_emb),
                         ('padder', self.padder),
                         ('rnn', rnn)])
        assert hasattr(pipe, 'predict')

    def test_remove_elements(self):
        pipe = Pipeline([('prepro', self.preprocessor),
                         ('vectors', self.doc_emb)])
        pipe.steps.pop()
        assert np.array_equal(pipe.transform(self.valid_input),
                              [['this', 'friend', 'angry'], ['hello', 'angry']])


if __name__ == '__main__':
    unittest.main()
