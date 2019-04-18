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
    """



    Parameters
    ----------
    vector_size : int (default=100)
        Number of dimensions of the document vector representation will have.

    window : int (default=5)
        Number of neighbouring words used to learn the vector representations.

    min_count : int (default=5)
        Minimum number of occurrences of a word in the training data in
        order to be added to the learned vocabulary.

    max_vocab_size : int (default=None)
        Maximum size of the vocabulary learned. Most infrequent words will
        not be added to the vocab in order to fulfill this constraint. If None,
        all the words that have at least min_count occurrences will be added
        to the vocabulary.
    """
    def __init__(self, size=100, alpha=0.025, window=5, min_alpha=0.0001,
                 min_count=5, max_vocab_size=None, sample=0.001,
                 seed=42, workers=3, epochs=100, hashfxn=hash):
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.iter = epochs
        self.hashfxn = hashfxn

    def to_doc2vec(self):
        params_dict = vars(self)
        params_dict['vector_size'] = params_dict.pop('size')
        params_dict['epochs'] = params_dict.pop('iter')
        return params_dict

    def to_word2vec(self):
        return vars(self)


class Doc2VecEmbedding(BaseEmbedding):
    """

    Parameters
    ----------


    model : Doc2Vec model (default=None)
        Doc2Vec loaded model to use. If None, a new model will be trained
        with the given GensimConfig. If you pass a model, the training step
        will be skipped, so the Gensim configuration will not be used.

    """

    def __init__(self, gensim_conf=GensimConfig(), model=None):
        self.model = model
        self.conf = gensim_conf

    def train(self, train_data):
        if self.model is None:
            _assert_valid_train_data(train_data)
            _assert_valid_conf(self.conf)
            tagged_data = [TaggedDocument(words=sent, tags=[idx]) for idx, sent
                           in enumerate(train_data)]
            self.model = Doc2Vec(documents=tagged_data,
                                 **self.conf.to_doc2vec())

    def to_vector(self, text):
        if self.model is None:
            raise NotFittedError
        return self.model.infer_vector(text)


class BaseWordEmbedding(BaseEmbedding, ABC):
    @abstractmethod
    def to_id(self, token):
        pass

    @abstractmethod
    def to_word(self, token_id):
        pass

    @abstractmethod
    def get_vectors(self):
        pass


class Word2VecEmbedding(BaseWordEmbedding):

    def __init__(self, gensim_conf=GensimConfig(), model=None):
        self.conf = gensim_conf
        self.model = model

    def train(self, train_data):
        """

        Parameters
        ----------

        train_data : iterable list of tokens
            Sentences used to train the word embedding.

        Returns
        -------

        """
        if self.model is None:
            _assert_valid_train_data(train_data)
            _assert_valid_conf(self.conf)
            self.model = Word2Vec(sentences=train_data,
                                  **self.conf.to_word2vec())

    def to_vector(self, token):
        return self.model.wv[token]

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

    def to_vector(self, token, **params):
        embeddings = self._get_embeddings_from(**params)
        index = embeddings.word2index[token]
        return embeddings.wv[index]

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
        self.embeddings_dict[language] = self.LoadedEmbedding(wv, index2word,
                                                              word2index)

    def _get_embeddings_from(self, **params):
        if params.get('language') is None:
            logger.warning('Language parameter was not specified. Input '
                           'sentences are assumed to be in English.')
            language = 'en'
        else:
            language = params['language']

        if self.embeddings_dict.get(language) is None:
            logger.info("Embeddings not available in cache, loading them...")
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

    class LoadedEmbedding:
        def __init__(self, wv, index2word, word2index):
            self.wv = wv
            self.index2word = index2word
            self.word2index = word2index


def _assert_valid_train_data(train_data):
    if not hasattr(train_data, '__iter__') or type(train_data) == str:
        raise InvalidArgumentError('train_data', 'Train data must be an '
                                                 'iterable of tokens.')
    elif len(train_data) == 0:
        raise InvalidArgumentError('train_data', "Train data can't be empty")


def _assert_valid_conf(gensim_conf):
    if not isinstance(gensim_conf, GensimConfig):
        raise InvalidArgumentError(gensim_conf, 'Configuration parameter '
                                   'must be an instance of the GensimConfig '
                                   'class.')
    elif gensim_conf.iter <= 0:
        raise InvalidArgumentError('epochs', 'Number of epochs must be greater '
                                             'than 0.')
    elif gensim_conf.window <= 0:
        raise InvalidArgumentError('window', 'Window size must be greater '
                                             'than 0.')
    elif gensim_conf.max_vocab_size is not None \
            and gensim_conf.max_vocab_size <= 0:
        raise InvalidArgumentError('max_vocab_size',
                                   'Maximum vocabulary size must be greater '
                                   'than 0.')
    elif gensim_conf.size <= 0:
        raise InvalidArgumentError('size', 'Vector size must be greater '
                                           'than 0.')
