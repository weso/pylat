import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, transformers):
        """

        :param transformers: List of transform functions that generate
        new features for the model.
        """
        self.transformers = transformers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        new_features = []
        for transformer in self.transformers:
            new_feature = []
            for sentence in X:
                new_feature.append(transformer(sentence))
            new_features.append(new_feature)
        return np.asarray(new_features).T.astype(float)
