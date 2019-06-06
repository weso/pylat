import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from pylat.exceptions import InvalidArgumentError


class SentencePadder(BaseEstimator, TransformerMixin):
    """Pads sentences of word vectors for use in a recurrent neural network.

    After preprocessing textual data and creating a vector representation of
    each word, each input will have different dimensions (one per word). In
    order to use the data in a recurrent neural network we need to pad every
    training instance with zeros up to a certain dimension. This dimension can
    be the greatest of the training instances or another one.

    Parameters
    ----------
    padding_length: int (default=None)
        Output length of each sample instance. If set to None, the output
        length will be equal to the max length from the sample set.

    Examples
    --------
    >>> from pylat.wrapper.transformer.sentence_padder import SentencePadder
    >>> X = [
    ...         [1, 4, 5, 2, 3, 5, 5, 1, 3],
    ...         [1, 2, 3],
    ...         [4, 5, 1, 2, 3, 1]
    ...     ]
    >>> default_padder = SentencePadder()
    >>> default_padder.fit_transform(X)
        array([[1., 4., 5., 2., 3., 5., 5., 1., 3.],
               [1., 2., 3., 0., 0., 0., 0., 0., 0.],
               [4., 5., 1., 2., 3., 1., 0., 0., 0.]], dtype=float32)
    >>> custom_padder = SentencePadder(padding_length=15)
    >>> custom_padder.fit_transform(X)
        array([[1., 4., 5., 2., 3., 5., 5., 1., 3., 0., 0., 0., 0., 0., 0.],
               [1., 2., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [4., 5., 1., 2., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)
    """

    def __init__(self, padding_length=None):
        self.padding_length = padding_length

    def fit(self, x, y=None):
        """Fits the padder to the given data.

        Parameters
        ----------
        x : :obj:`list` of :obj:`list`
            List of lists with the numerical representation of each sentence.
        y : :obj:`list`, optional (default=None)
            List of labels of each sentence.

        Returns
        -------
        self
            Reference to the class after being trained.

        Raises
        ------
        InvalidArgumentError
            If the padding length is lower than the maximum length of the
            sentences in the given array.
        """
        max_length = len(max(x, key=len))
        if self.padding_length is not None and self.padding_length < max_length:
            raise InvalidArgumentError('padding_length',
                                       'Padding length must be greater or \
                                       equal to the maximum sentence length.')
        elif self.padding_length is None:
            self.padding_length = max_length
        return self

    def transform(self, x):
        """Return the padded array.

        Parameters
        ----------
        x : :obj:`list` of :obj:`list`
            List of lists with the numerical representation of each sentence.

        Returns
        -------
        :obj:`np.array`
            Padded array with as many columns as the given padding length.
        """
        num_instances = len(x)
        ret = np.zeros(shape=[num_instances, self.padding_length],
                       dtype=np.float32)
        for idx, sentence in enumerate(x):
            if len(sentence) != 0:
                shortened_sentence = sentence[:self.padding_length]
                ret[idx, :len(shortened_sentence)] = shortened_sentence
        return ret
