from sklearn.base import BaseEstimator, TransformerMixin


class ToArrayTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()
