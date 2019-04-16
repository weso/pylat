from sklearn.base import BaseEstimator, TransformerMixin


class WordEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embeddings, fit_corpus=None, to_id=True):
        pass

    def fit(self, x, y=None, **fit_params):
        pass


class DocumentEmbeddingsTransformer():
    def __init__(self, embeddings, fit_corpus=None):
        pass