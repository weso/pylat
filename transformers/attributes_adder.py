import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


def test_transformer(X):
    """ Transformer example

    :param X: Pandas DataFrame containing the training data
    :return: Array containing the id of each post multiplied by 2.
    """
    return X[:, id] * 2


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
        for transformer in self.transformers:
            X = np.c_(transformer(X))
        return X