from sklearn.base import TransformerMixin

import numpy as np


class WordEmbeddingsTransformer(TransformerMixin):
    """Transform a list of tokens to numbers using a word embedding.

    This is a wrapper around word2vec, complying to the scikit-learn
    transformer API, which can be used in a transformer pipeline to
    transform some input textual data to word vectors.

    The word vectors are learned based on the parameters given
    in the constructor of the embedding, allowing its use for hyper-parameter
    tuning (e.g. RandomizedSearchCV).

    Parameters
    ----------
    embeddings : :obj:`BaseWordEmbedding`
        Instance of BaseWordEmbedding class that will be used to train the
        embeddings and transform the tokens.
    fit_corpus : :obj:`list` of :obj:`list` of str, optional (default=None)
        A 2 dimensional list, where the first dimension contains every sentence
        used to train the embedding, and the second dimension contains every
        token of each sentence. If set to None, the sentences to be transformed
        will be fed as training data to the embedding.
    to_id : bool, optional (default=True)
        If set to True, each token will be transformed to their id in the
        embedding. If set to False, each token will be transformed to their
        vector representation instead.

    Examples
    --------
    >>> from pylat.wrapper.transformer.embeddings_transformer import \
            WordEmbeddingsTransformer
    >>> from pylat.neuralnet.embeddings import GensimConfig, Word2VecEmbedding
    >>> X = [['Sample', 'text'], ['another', 'one'], ['last', 'one']]
    >>> # arguments passed in order to make execution deterministic
    >>> config = GensimConfig(size=2, epochs=5, workers=1, seed=42, \
                              min_count=1, window=1, hashfxn=len)
    >>> embedding = Word2VecEmbedding(config)
    >>> vec_transformer = WordEmbeddingsTransformer(embedding, to_id=False)
    >>> vec_transformer.fit(X).transform(X)
        array([[[ 0.18671471,  0.23427033], [ 0.19643007, -0.08401009]],
               [[-0.24481292,  0.0009373 ], [-0.13900341,  0.18536615]],
               [[ 0.19643007, -0.08401009], [-0.13900341,  0.18536615]]],
               dtype=float32)
    >>> id_transformer = WordEmbeddingsTransformer(embedding)
    >>> id_transformer.fit(X).transform(X)
        array([[1, 2],
               [3, 0],
               [4, 0]])
    """

    def __init__(self, embeddings, fit_corpus=None, to_id=True):
        self.embeddings = embeddings
        self._fit_corpus = fit_corpus
        self._to_id = to_id

    def fit(self, x, y=None, **fit_params):
        """Fits the word embeddings to the given data

        Note
        ----
        If a fit_corpus was passed in the constructor of this class, that data
        will be used to fit the embeddings instead.

        Parameters
        ----------
        x : :obj:`list` of :obj:`list` of str
            A 2 dimensional list, where the first dimension contains every
            sentence used to train the embedding, and the second dimension
            contains every token of each sentence. If set to None, the sentences
            to be transformed will be fed as training data to the embedding.
        y : :obj:`list`, optional (default=None)
            Labels of the passed data.

        Returns
        -------
        self
            Reference to the class after being trained.
        """
        train_data = x if self._fit_corpus is None else self._fit_corpus
        self.embeddings.train(train_data, **fit_params)
        return self

    def transform(self, x, **transf_params):
        """Transforms the input text into numbers.

        This method uses the embeddings trained in the fit method to obtain
        the vector representation of the given texts. A vector will be returned
        for each token in the texts, and these tokens combined compose the
        numerical representation of each text.

        Parameters
        -----------
        x : :obj:`list` of :obj:`list` of str
            2 dimensional list, containing the tokens of every sentence.

        Returns
        -------
        return : numpy array
            Each row in the output array corresponds to each document in the
            input array. If to_id was set to True, the second dimension will
            contain the id of each token in the embedding. If it was set to
            false, it will contain the embedding of each token.
        """
        func = self.embeddings.to_id if self._to_id \
            else self.embeddings.to_vector
        return np.asarray([[func(token, **transf_params) for token in sentence
                           if func(token, **transf_params) is not None]
                           for sentence in x])

    def fit_transform(self, X, y=None, **params):
        return self.fit(X, **params).transform(X, **params)


class DocumentEmbeddingsTransformer(TransformerMixin):
    """Transforms a list of sentences into their corresponding embeddings.

    This is a wrapper around doc2vec, complying to the scikit-learn
    transformer API, which can be used in a transformer pipeline to
    transform some input textual data to document vectors.

    The document vectors are learned based on the parameters given
    in the constructor, allowing its use for hyper-parameter tuning
    (e.g. RandomizedSearchCV).

    Parameters
    ----------
    embeddings : :obj:`Doc2VecEmbedding`
        Instance of the Doc2VecEmbedding class used to transform the sentences.
    fit_corpus : :obj:`list` of :obj:`list` of str, optional (default=None)
        A 2 dimensional list, containing the tokens of every sentence that will
        be used to train the embeddings. If set to None, the data to be
        transformed into vectors will be also used as training data.

    Examples
    --------
    >>> from pylat.wrapper.transformer.embeddings_transformer import \
            DocumentEmbeddingsTransformer
    >>> from pylat.neuralnet.embeddings import GensimConfig, Doc2VecEmbedding
    >>> X = [['Sample', 'text'], ['another', 'one'], ['last', 'one']]
    >>> # arguments passed in order to make execution deterministic
    >>> config = GensimConfig(size=2, epochs=5, workers=1, seed=42, \
                              min_count=1, window=1, hashfxn=len)
    >>> embedding = Doc2VecEmbedding(config)
    >>> d2v = DocumentEmbeddingsTransformer(embedding, fit_corpus=X)
    >>> d2v.fit(X).transform(X)
    array([[-0.15986516, -0.24026237],
           [-0.15986516, -0.24026237],
           [ 0.18679854,  0.23427881]], dtype=float32)
    """

    def __init__(self, embeddings, fit_corpus=None):
        self.embeddings = embeddings
        self._fit_corpus = fit_corpus

    def fit(self, x, y=None):
        """Fits the document embeddings to the given data

        Note
        ----
        If a fit_corpus was passed in the constructor of this class, that data
        will be used to fit the embeddings instead.

        Parameters
        ----------
        x : :obj:`list` of :obj:`list` of str
            A 2 dimensional list, where the first dimension contains every
            sentence used to train the embedding, and the second dimension
            contains every token of each sentence. If set to None, the sentences
            to be transformed will be fed as training data to the embedding.
        y : :obj:`list`, optional (default=None)
            Labels of the passed data.

        Returns
        -------
        self
            Reference to the class after being trained.
        """
        train_data = x if self._fit_corpus is None else self._fit_corpus
        self.embeddings.train(train_data)
        return self

    def transform(self, x):
        """Transforms the input text into numbers.

        This method uses the embeddings trained in the fit method to obtain a
        vector representation of each given text.

        Parameters
        -----------
        x : :obj:`list` of :obj:`list` of str
            2 dimensional list, containing the tokens of every sentence.

        Returns
        -------
        return : numpy array of shape of shape [n, m]
            Each row in the output array corresponds to each document in the
            input array, and it contains as many numbers ('m') as the dimension
            of the embedding passed as a parameter.
        """
        return np.asarray([self.embeddings.to_vector(sentence)
                           for sentence in x])
