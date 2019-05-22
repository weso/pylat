from pylat.neuralnet.embeddings import GensimConfig, Word2VecEmbedding, \
    Doc2VecEmbedding, CrossLingualPretrainedEmbedding
from pylat.neuralnet.utils import convert_vec_embedding
from pylat.exceptions import InvalidArgumentError
from gensim.models import Doc2Vec, Word2Vec

import numpy as np
import os
import pytest
import unittest


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        self.valid_input = [
            ["This", "is", "the", "first", "sentence"],
            ["And", "this", "is", "the", "second", "one"],
            ["Each", "sentence", "is", "a", "list", "of", "tokens"]
        ]
        self.invalid_input = "This is some text"
        self.embeddings_dir = os.path.join('test', 'data', 'embeddings')

    def test_pretrained_word2vec(self):
        model = Word2Vec.load(os.path.join(self.embeddings_dir,
                                           'w2v_test.model'))
        embeddings = Word2VecEmbedding(model=model)
        embeddings.train(None)  # shouldn't do anything in train
        self.assertEqual(embeddings.model, model)
        self.assertEqual(embeddings.to_id('pron'), 0)
        self.assertEqual(embeddings.to_id('invented'), None)

    def test_invalid_conf(self):
        invalid_configs = [GensimConfig(epochs=0), GensimConfig(window=0),
                           GensimConfig(size=0), GensimConfig(max_vocab_size=0)]
        for config in invalid_configs:
            embeddings = Word2VecEmbedding(gensim_conf=config)
            with pytest.raises(InvalidArgumentError):
                embeddings.train(self.valid_input)

    def test_to_id(self):
        config = GensimConfig(epochs=5, max_vocab_size=None, min_count=1)
        embeddings = Word2VecEmbedding(gensim_conf=config)
        embeddings.train(self.valid_input)
        self.assertEqual(embeddings.to_id('sentence'), 2)
        self.assertEqual(embeddings.to_word(2), 'sentence')

    def test_to_vector(self):
        config = GensimConfig(epochs=5, max_vocab_size=None,
                              min_count=1, size=5)
        w2v = Word2VecEmbedding(gensim_conf=config)
        w2v.train(self.valid_input)
        self.assertEqual(len(w2v.to_vector('sentence')), 5)
        d2v = Doc2VecEmbedding(gensim_conf=config)
        d2v.train(self.valid_input)
        self.assertEqual(len(d2v.to_vector(['my', 'sentence'])), 5)

    def test_pretrained_doc2vec(self):
        model = Doc2Vec.load(os.path.join(self.embeddings_dir,
                                          'd2v_test.model'))
        embeddings = Doc2VecEmbedding(model=model)
        embeddings.train(None)  # shouldn't do anything in train
        self.assertEqual(embeddings.model, model)

    def test_spanish_multilingual_embedding(self):
        model = CrossLingualPretrainedEmbedding(self.embeddings_dir)
        model.set_language('es')
        self.assertEqual(model.to_id('amigo'), 0)
        np.testing.assert_allclose(model.to_vector('amor'),
                                   np.asarray([0.7, 0.2, 0.4]))

    def test_change_language_multilingual_embedding(self):
        model = CrossLingualPretrainedEmbedding(self.embeddings_dir)
        model.set_language('es')
        self.assertEqual(model.to_id('tierno'), 4)
        model.set_language('en')
        self.assertEqual(model.to_id('tierno'), None)
        np.testing.assert_allclose(model.to_vector('cute'),
                                   np.asarray([0.3, 0.8, 0.6]))

    def test_no_language_set_multilingual_embedding(self):
        model = CrossLingualPretrainedEmbedding(self.embeddings_dir)
        self.assertEqual(model.to_id('amigo'), None)
        self.assertEqual(model.to_id('friend'), 2)

    def test_invalid_language_multilingual_embedding(self):
        model = CrossLingualPretrainedEmbedding(self.embeddings_dir)
        model.set_language('ko')
        with pytest.raises(InvalidArgumentError):
            model.to_id('친구')

    def test_save_embeddings(self):
        conf = GensimConfig(size=2, min_count=1)
        embeddings = Word2VecEmbedding(conf)
        embeddings.train(self.valid_input)
        embeddings.save_embeddings('emb')
        emb_arr = np.load('emb.npy')
        self.assertEqual(np.shape(emb_arr), (14, 2))
        with open('emb.vocab', 'r') as f:
            vocab = f.readlines()
            self.assertEqual(len(vocab), 14)
        os.remove('emb.npy')
        os.remove('emb.vocab')

    def test_invalid_input(self):
        conf = GensimConfig(min_count=1)
        embeddings = Word2VecEmbedding(conf)
        with pytest.raises(InvalidArgumentError):
            embeddings.train(self.invalid_input)

    def test_embedding_conversion(self):
        convert_vec_embedding(os.path.join(self.embeddings_dir, 'de.vec'),
                              self.embeddings_dir)
        vectors = np.load(os.path.join(self.embeddings_dir, 'de.npy'))
        assert np.allclose(vectors, [[-0.011, -0.002, -0.051, 0.020, -0.069],
                                     [-0.046, -0.000, -0.075, 0.027, -0.101],
                                     [-0.032, -0.046, -0.008, 0.099, 0.014]],
                           atol=1e-3)
        with open(os.path.join(self.embeddings_dir, 'de.vocab'), 'r') as f:
            vocab = f.readlines()
            assert sorted(vocab) == sorted([',\n', '.\n', 'gut\n'])
        os.remove(os.path.join(self.embeddings_dir, 'de.npy'))
        os.remove(os.path.join(self.embeddings_dir, 'de.vocab'))


if __name__ == '__main__':
    unittest.main()
