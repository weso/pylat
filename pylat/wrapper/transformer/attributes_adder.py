import numpy as np

from pylat.exceptions import InvalidArgumentError
from sklearn.base import BaseEstimator, TransformerMixin


class AttributesAdder(BaseEstimator, TransformerMixin):
    """Add multiple attributes to a machine learning pipeline.

    This class allows the addition of multiple attributes to a text pipeline.
    It can be used, for example, to join the vector representation of a document
    and its valence score into a single vector that will be fed to a learning
    algorithm.

    Parameters
    ----------
    transformers : :obj:`list` of callable
        Iterable of callables that will be called to obtain the new combined
        features. The callables must receive a sentence (iterable of tokens),
        and return its corresponding numerical representation.
    """

    def __init__(self, transformers):
        self.transformers = transformers
        self._assert_valid_input()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Return the vector representation of the input texts.

        Parameters
        ----------
        X : :obj:`list` of :obj:`list` of str
            List of lists, where the first dimension will contain every sentence
            that will be transformed, and each sentence will contain a list of
            every token.
        y : :obj:`list`, optional (default=None)
            List of labels of the input texts. This labels will not be used
            in the transformation step.

        Returns
        -------
        return : numpy array
            Each row in the output array corresponds to each document in the
            input array, and contains as many elements as new features have
            been obtained from the transformers.
        """
        new_features = []
        for transformer in self.transformers:
            new_feature = []
            for sentence in X:
                new_feature.append(transformer(sentence))
            new_features.append(new_feature)
        return np.asarray(new_features).T.astype(float)

    def _assert_valid_input(self):
        if not isinstance(self.transformers, list):
            raise InvalidArgumentError(self.transformers, "Transformers param "
                                                          "must be a list.")

        for tr in self.transformers:
            if not callable(tr):
                raise InvalidArgumentError(tr, "All transformers must be "
                                           "callables.")
