from gensim.models import Word2Vec
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from .exceptions import InvalidArgumentError

import numpy as np


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """ Sklearn transformer that creates word vectors from textual data.

    This is a wrapper around word2vec, complying to the scikit-learn
    transformer API, which can be used in a transformer pipeline to
    transform some input textual data to word vectors.

    The word vectors are learned based on the parameters given
    in the constructor, allowing its use for hyper-parameter tuning
    (e.g. RandomizedSearchCV).

    Parameters
    ----------
    word2vec_class: BaseWordEmbeddingsModel class (default=Word2Vec)
        Class used for creating the word vectors.

    fit_corpus: array of sentences (default=None)
        Sentences used to train the word embedding. If None, the input data
        will be used to train them.

    size: int (default=100)
        Number of dimensions of the word vector representation.

    window: int (default=5)
        Number of neighbouring words used to learn the vector representations.

    min_count: int (defalt=5)
        Minimum number of occurrences of a word in the training data in
        order to be added to the learned vocabulary.

    max_vocab_size: int (default=None)
        Maximum size of the vocabulary learned. Most infrequent words will
        not be added to the vocab in order to fulfill this constraint. If None,
        all the words that have at least min_count occurrences will be added
        to the vocabulary.

    Examples
    --------
    >>> from src.transformer.word2vec_transformer import Word2VecTransformer
    >>> X = [['Sample', 'text'], ['another', 'one'], ['last', 'one']]
    >>> # arguments passed to transformer in order to make execution deterministic
    >>> w2v = Word2VecTransformer(size=2, iters=5, workers=1, random_seed=42, min_count=1, window=1, hashfxn=len)
    >>> w2v.fit(X).transform(X)
        array([[[ 0.18671471,  0.23427033], [ 0.19643007, -0.08401009]],
               [[-0.24481292,  0.0009373 ], [-0.13900341,  0.18536615]],
               [[ 0.19643007, -0.08401009], [-0.13900341,  0.18536615]]], dtype=float32)
    """

    def __init__(self, word2vec_class=Word2Vec, fit_corpus=None, size=100, alpha=0.025,
                 window=5, min_count=5, max_vocab_size=None, sample=0.001, random_seed=42,
                 workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75,
                 cbow_mean=1, hashfxn=hash, iters=100, null_word=0, trim_rule=None, sorted_vocab=1,
                 batch_words=10000, compute_loss=False, max_final_vocab=None):
        self.model = None
        self.fit_corpus = fit_corpus
        self.word2vec_class = word2vec_class
        self.missed_tokens = 0
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.random_seed = random_seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.cbow_mean = cbow_mean
        self.hashfxn = hashfxn
        self.iters = iters
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.compute_loss = compute_loss
        self.max_final_vocab = max_final_vocab

    def fit(self, x, y=None, **fit_params):
        self._assert_valid_params()

        if self.fit_corpus is None:
            self.fit_corpus = x

        self.model = Word2Vec(
            self.fit_corpus, size=self.size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, seed=self.random_seed,
            workers=self.workers, min_alpha=self.min_alpha, sg=self.sg, hs=self.hs,
            negative=self.negative, ns_exponent=self.ns_exponent, cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn, iter=self.iters, null_word=self.null_word,
            trim_rule=self.trim_rule, sorted_vocab=self.sorted_vocab, batch_words=self.batch_words,
            compute_loss=self.compute_loss, max_final_vocab=self.max_final_vocab
        )
        return self

    def transform(self, x):
        if self.model is None:
            raise NotFittedError

        self.missed_tokens = 0
        ret = []
        for sentence in x:
            ret.append(np.asarray([self._get_vector_for(token) for token in sentence
                                   if self._get_vector_for(token) is not None]))
        return np.asarray(ret)

    def _assert_valid_params(self):
        if not issubclass(self.word2vec_class, BaseWordEmbeddingsModel):
            raise InvalidArgumentError('word2vec_class',
                                       'Word2Vec class must subclass BaseWordEmbeddingsModel from gensim')

    def _get_vector_for(self, token, default=None):
        try:
            return self.model.wv[token]
        except KeyError:
            self.missed_tokens += 1
            return default
