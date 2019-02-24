from gensim.models import Doc2Vec
from gensim.models.base_any2vec import BaseWordEmbeddingsModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from src.exceptions import InvalidArgumentError

import numpy as np


class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    """ Sklearn transformer that creates document vectors from textual data.

    This is a wrapper around doc2vec, complying to the scikit-learn
    transformer API, which can be used in a transformer pipeline to
    transform some input textual data to document vectors.

    The document vectors are learned based on the parameters given
    in the constructor, allowing its use for hyper-parameter tuning
    (e.g. RandomizedSearchCV).

    Parameters
    ----------
    fit_corpus: iterable list of TaggedDocument
        Sentences used to train the word embedding.

    model: Doc2Vec model (default=None)
        Doc2Vec loaded model to use. If None, a new model will be trained
        with the rest of the params. If you pass a model, the training step
        will be skipped, so the rest of the params will not be used.

    vector_size: int (default=100)
        Number of dimensions of the document vector representation will have.

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
    >>> from src.transformers.doc2vec_transformer import Doc2VecTransformer
    >>> from gensim.models.doc2vec import TaggedDocument, Doc2Vec
    >>> X = [['Sample', 'text'], ['another', 'one'], ['last', 'one']]
    >>> tagged_data = [TaggedDocument(words=text, tags=[str(idx)]) for idx, text in enumerate(X)]
    >>> # arguments passed to transformer in order to make execution deterministic
    >>> d2v = Doc2VecTransformer(fit_corpus=tagged_data, size=2, epochs=5, workers=1, random_seed=42, min_count=1, window=1, hashfxn=len)
    >>> d2v.fit(X).transform(X)
    array([[-0.15986516, -0.24026237],
           [-0.15986516, -0.24026237],
           [ 0.18679854,  0.23427881]], dtype=float32)
    """

    def __init__(self, fit_corpus, model=None, size=100, alpha=0.025,
                 window=5, min_alpha=0.0001, min_count=5, max_vocab_size=None, sample=0.001, random_seed=42,
                 workers=3, epochs=100, hashfxn=hash):
        self.model = None
        self.fit_corpus = fit_corpus
        self.missed_tokens = 0
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.random_seed = random_seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.epochs = epochs
        self.hashfxn = hashfxn

    def fit(self, x, y=None, **fit_params):
        # if we already have a model we skip the training step
        if self.model is not None:
            return self

        self.model = Doc2Vec(
            self.fit_corpus, size=self.size, alpha=self.alpha, window=self.window,
            min_count=self.min_count, max_vocab_size=self.max_vocab_size, seed=self.random_seed,
            workers=self.workers, hashfxn=self.hashfxn, min_alpha=self.min_alpha, epochs=self.epochs
        )
        return self

    def transform(self, x):
        if self.model is None:
            raise NotFittedError

        ret = []
        for sentence in x:
            ret.append(self.model.infer_vector(sentence))
        return np.asarray(ret)
