from sklearn.base import BaseEstimator, TransformerMixin


class ToArrayTransformer(BaseEstimator, TransformerMixin):
    """Returns the array representation of the given list.

    This class is meant to be incorporated to a sklearn pipeline to convert
    an iterable to a numpy array that can be used later on by the machine
    learning algorithms

    Examples
    --------
    >>> from pylat.wrapper.transformer import ToArrayTransformer
    >>> from scipy.sparse import dok_matrix
    >>> X = dok_matrix([[1, 0], [0, 1]])
    >>> toarray = ToArrayTransformer()
    >>> print(toarray.fit_transform(X))
    [[1 0]
    [0 1]]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()
