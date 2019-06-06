from abc import ABC, abstractmethod
from pylat.exceptions import InvalidArgumentError
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
    """Base class of all the embeddings.

    This class provides the two methods that all the embedding classes provide
    to the user. It can be extended by other classes to add extra functionality.
    """

    @abstractmethod
    def train(self, train_data):
        """Train the embedding with the given training data.

        Parameters
        ----------
        train_data

        Returns
        -------
        self
            Trained embedding instance.
        """
        pass

    @abstractmethod
    def to_vector(self, text):
        """Transforms text into its numerical representation

        Parameters
        ----------
        text

        Returns
        -------
        :obj:`list` of float
            Vector representation of the input text.
        """
        pass


class GensimConfig:
    """ Configuration class for embeddings from the gensim module.

    This class stores the parameters needed to train embeddings from the
    gensim module ('word2vec' and 'doc2vec'). It is used by the
    Word2VecEmbedding and Doc2VecEmbedding classes in order to train their
    respective models.

    Parameters
    ----------
    epochs : int, optional (default=100)
        Number of epochs used to learn the vector representations.

    size : int, optional (default=100)
        Number of dimensions that the final vector representation will have.

    min_count : int, optional (default=5)
        Minimum number of occurrences of a word in the training data in
        order to be added to the learned vocabulary.

    max_vocab_size : int, optional (default=None)
        Maximum size of the vocabulary learned. Most infrequent words will
        not be added to the vocab in order to fulfill this constraint. If None,
        all the words that have at least min_count occurrences will be added
        to the vocabulary.

    window : int, optional (default=5)
        Number of neighbouring words used to learn the vector representations.

    alpha : float, optional (default=0.025)
        Initial learning rate of the model.

    min_alpha : float, optional (default=0.0001)
        Minimum learning rate that the model will reach during training.

    sample : float, optional (default=0.01)
        This value configures which words with higher frequency are downsampled.

    seed : int, optional (default=42)
        Random seed used internally by the model in the training phase.

    workers : int, optional (default=3)
        Number of threads used to train the model.

    hashfxn : func, optional (default=hash)
        Function used internally to perform hashing of the words. If you want
        to obtain completely deterministic results, you can set this function
        to `len`. Otherwise, it is recommended to leave the default value.
    """
    def __init__(self, epochs=100, size=100, min_count=5, max_vocab_size=None,
                 window=5, alpha=0.025, min_alpha=0.0001, sample=0.001,
                 seed=42, workers=3, hashfxn=hash):
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
        """Transforms the configuration to the specific Doc2Vec class names.

        Returns
        -------
        dict
            Dictionary containing every parameter of the configuration used
            by the Doc2Vec Gensim class.
        """
        params_dict = vars(self)
        # change the following two params to be the same as the Doc2Vec ones
        params_dict['vector_size'] = params_dict.pop('size')
        params_dict['epochs'] = params_dict.pop('iter')
        return params_dict

    def to_word2vec(self):
        """Transforms the configuration to the specific Word2Vec class names.

        Returns
        -------
        dict
            Dictionary containing every parameter of the configuration used
            by the Word2Vec Gensim class.
        """
        return vars(self)


class Doc2VecEmbedding(BaseEmbedding):
    """Embedding class that transforms sentences into vectors.

    This class is a wrapper around the Gensim Doc2Vec implementation that
    implements the BaseEmbedding interface.

    Parameters
    ----------
    gensim_conf : :obj:`GensimConfig` (default=GensimConfig())
        An instance of the GensimConfig class providing the parameters that will
        be used to train the embeddings.

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
    """Base class of all the word embedding classes.

    This class provides the basic interface of all the embeddings that work at
    a word level.
    """

    @abstractmethod
    def to_id(self, token):
        """Return the id of the given token in the embedding.

        Parameters
        ----------
        token : str
            Token for which the id will be retrieved.

        Returns
        -------
        int
            ID of the token if it is present in the embedding, None otherwise.
        """
        pass

    @abstractmethod
    def to_word(self, token_id):
        """Returns the word corresponding to the given id in the embedding.

        Parameters
        ----------
        token_id : int
            ID of the token that will be retrieved.

        Returns
        -------
        str
            Token with the given id if it is present in the embedding,
            None otherwise.
        """
        pass

    @abstractmethod
    def get_vectors(self):
        """Returns all the vectors computed by the embedding instance.

        Returns
        -------
        2d array of float
            Values of the numeric representations of every word.
        """
        pass


class Word2VecEmbedding(BaseWordEmbedding):
    """Class that transforms words into vectors using Word2Vec."""

    def __init__(self, gensim_conf=GensimConfig(), model=None):
        self.conf = gensim_conf
        self.model = model

    def train(self, train_data):
        if self.model is None:
            _assert_valid_train_data(train_data)
            _assert_valid_conf(self.conf)
            self.model = Word2Vec(sentences=train_data,
                                  **self.conf.to_word2vec())

    def to_vector(self, token):
        try:
            return self.model.wv[token]
        except KeyError:
            return None

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
        """Save the embeddings to a given directory.

        The embeddings will be saved in two different files. A .vocab file that
        will hold all of the vocabulary learnt by the Word2Vec model in a list
        of words. The second one is a .npy file that store the weights of each
        word.

        Parameters
        ----------
        save_path : str
            Directory where the embedding will be saved.
        """
        np.save(save_path + '.npy', self.model.wv.vectors)
        with codecs.open(save_path + '.vocab', 'w', 'utf-8') as f_out:
            for word in self.model.wv.index2word:
                f_out.write(word + '\n')


class CrossLingualPretrainedEmbedding(BaseWordEmbedding):
    """Class that transforms words into vectors using pretrained embeddings.

    This class is meant to be used with embeddings for several languages
    aligned in a common vector space. All of the embeddings must belong to the
    same folder. The language used by the class can be changed at runtime.

    Parameters
    ----------
    embeddings_dir : str
        Directory where the pretrained embeddings are located.
    language : str, optional (default=None)
        Language to use by the class. If set to None, English language will be
        used by default.
    """

    def __init__(self, embeddings_dir, language=None):
        self.embeddings_dir = embeddings_dir
        self.embeddings_dict = {}
        self.language = language

    def train(self, train_data):
        pass

    def to_vector(self, token):
        embeddings = self._get_embeddings()
        try:
            index = embeddings.word2index[token]
            return embeddings.wv[index]
        except KeyError:
            return None

    def set_language(self, language):
        """Change the language used by the embeddings internally.

        Parameters
        ----------
        language : str
            Language to use by the class.
        """
        self.language = language

    def to_id(self, token):
        try:
            return self._get_embeddings().word2index[token]
        except KeyError:
            return None

    def to_word(self, token_id):
        return self._get_embeddings().index2word[token_id]

    def get_vectors(self):
        embeddings = self._get_embeddings()
        return embeddings.wv

    def _get_embeddings(self):
        if self.language is None:
            logger.warning('Language attribute was not specified. Input '
                           'sentences are assumed to be in English.')
            language = 'en'
        else:
            language = self.language

        if self.embeddings_dict.get(language) is None:
            logger.info("Embeddings not available in cache, loading them...")
            self._load_vector(language)
        return self.embeddings_dict[language]

    def _load_vector(self, language):
        vector_path = os.path.join(self.embeddings_dir, language)
        self._assert_files_exist(vector_path, language)
        with codecs.open(vector_path + '.vocab', 'r', 'utf-8') as f_in:
            index2word = [line.strip() for line in f_in]
        word2index = {w: i for i, w in enumerate(index2word)}
        wv = np.load(vector_path + '.npy')
        self.embeddings_dict[language] = self.LoadedEmbedding(wv, index2word,
                                                              word2index)

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
        """Inner class that stores information about the loaded embeddings.

        Parameters
        ----------
        wv : :obj:`np.array`
            Matrix with all the word vectors.
        index2word : :obj:`list` of str
            Ordered list with every word of the embeddings.
        word2index : dict
            Dictionary that maps every word to its internal index.
        """
        def __init__(self, wv, index2word, word2index):
            self.wv = wv
            self.index2word = index2word
            self.word2index = word2index


def _assert_valid_train_data(train_data):
    if not hasattr(train_data, '__iter__') or isinstance(train_data, str):
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
