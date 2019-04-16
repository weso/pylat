from abc import ABC, abstractmethod
from easynlp.exceptions import InvalidArgumentError
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from sklearn.exceptions import NotFittedError

import codecs
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseEmbedding(ABC):
    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def to_vector(self, text):
        pass


class GensimConfig():
    def __init__(self, size=100, alpha=0.025, window=5, min_alpha=0.0001,
                 min_count=5, max_vocab_size=None, sample=0.001,
                 random_seed=42, workers=3, epochs=100, hashfxn=hash):
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.random_seed = random_seed
        self.workers = workers
        self.epochs = epochs
        self.hashfxn = hashfxn


class Doc2VecEmbedding(BaseEmbedding):
    def __init__(self, gensim_conf, model=None):
        self.model = model
        self.conf = gensim_conf

    def train(self, train_data):
        if self.model is None:
            self._assert_valid_train_data(train_data)
            tagged_data = [TaggedDocument(words=sent, tags=[idx]) for idx, sent
                           in enumerate(train_data)]
            self.model = Doc2Vec(documents=tagged_data, **self.conf.vars())

    def to_vector(self, text):
        if self.model is None:
            raise NotFittedError
        return self.model.infer_vector(text)

    def _assert_valid_train_data(self, train_data):
        pass


class BaseWordEmbedding(BaseEmbedding, ABC):
    @abstractmethod
    def to_id(self, text):
        pass

    @abstractmethod
    def to_word(self, token_id):
        pass

    @abstractmethod
    def get_vectors(self):
        pass


class Word2VecEmbedding(BaseWordEmbedding):
    def __init__(self, gensim_conf, model=None):
        self.conf = gensim_conf
        self.model = model

    def train(self, train_data):
        if self.model is None:
            self._assert_valid_train_data(train_data)
            self.model = Word2Vec(sentences=train_data, **self.conf.vars())

    def to_vector(self, text):
        return np.asarray([self.model.infer_vector(word)
                           for word in text])

    def to_id(self, token):
        try:
            return self.model.wv.vocab[token].index
        except KeyError:
            return None

    def to_word(self, token_id):
        return self.model.wv.index2word[token_id]

    def get_vectors(self):
        return self.model.wv.vectors

    def save_embeddings(self, save_path):
        np.save(save_path + '.npy', self.model.wv.vectors)
        with codecs.open(save_path + '.vocab', 'w', 'utf-8') as f_out:
            for word in self.model.wv.index2word:
                f_out.write(word + '\n')


class CrossLingualPretrainedEmbedding(BaseWordEmbedding):
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.embeddings_dict = {}

    def train(self, train_data):
        pass

    def to_vector(self, text, **params):
        embeddings = self._get_embeddings_from(**params)
        return np.asarray([embeddings.query(word)
                           for word in text])

    def to_id(self, token, **params):
        return self._get_embeddings_from(**params).word2index[token]

    def to_word(self, token_id, **params):
        return self._get_embeddings_from(**params).index2word[token_id]

    def get_vectors(self, **params):
        embeddings = self._get_embeddings_from(**params)
        return embeddings.get_vectors_mmap()

    def _load_vector(self, language):
        vector_path = os.path.join(self.embeddings_dir, language)
        self._assert_files_exist(vector_path, language)
        with codecs.open(vector_path + '.vocab', 'r', 'utf-8') as f_in:
            index2word = [line.strip() for line in f_in]
        word2index = {w: i for i, w in enumerate(index2word)}
        wv = np.load(vector_path + '.npy')
        self.embeddings_dict[language] = (wv, index2word, word2index)

    def _get_embeddings_from(self, **params):
        if params.get('language') is None:
            logger.warning('Language parameter was not specified. Input '
                           'sentences are assumed to be in English.')
            language = 'en'
        else:
            language = params['language']

        if self.embeddings_dict.get(language) is None:
            logger.warning("Embeddings not available in cache, loading them...")
            self._load_vector(language)

        return self.embeddings_dict[language]

    def _assert_files_exist(self, vector_path, language):
        if not os.path.exists(vector_path + '.vocab'):
            raise InvalidArgumentError(self.embeddings_dir, 'Could not find '
                                       'vocab file for language {} in given '
                                       'directory.'.format(language))
        if not os.path.exists(vector_path + '.npy'):
            raise InvalidArgumentError(self.embeddings_dir, 'Could not find '
                                       'npy file for language {} in given '
                                       'directory.'.format(language))
