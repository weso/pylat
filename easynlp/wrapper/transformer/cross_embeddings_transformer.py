from easynlp.exceptions import InvalidArgumentError

from pymagnitude import Magnitude
from sklearn.base import BaseEstimator, TransformerMixin

import logging
import numpy as np
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CrossEmbeddingsTransformer(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.embeddings_dict = {}

    def fit(self, x, y=None, **fit_params):
        # TODO: load vectors lazily, when transform needs them
        self._load_vectors()
        return self

    def transform(self, x, **transform_params):
        """
        """
        if transform_params.get('language') is None:
            logger.warning('Language parameter was not specified. Input '
                           'sentences are assumed to be in English.')
            language = 'en'
        else:
            language = transform_params['language']

        if self.embeddings_dict.get(language) is None:
            logger.warning("Embeddings for language '%s' are not "
                           "available. Using English embeddings...")
            embeddings = self.embeddings_dict['en']
        else:
            embeddings = self.embeddings_dict[language]

        ret = []
        for sentence in x:
            ret.append(np.asarray([embeddings.query(word)
                                   for word in sentence
                                   if word in embeddings]))
        return np.asarray(ret)

    def _load_vectors(self):
        for filename in os.listdir(self.embeddings_dir):
            self._assert_valid_file(filename)
            language = filename.split('.')[-2]
            embeddings = Magnitude(os.path.join(self.embeddings_dir, filename))
            self.embeddings_dict[language] = embeddings

    def _assert_valid_file(self, file):
        tokens = file.split('.')
        if len(tokens) != 2 or tokens[-1] != 'magnitude':
            raise InvalidArgumentError(self.embeddings_dir, 'All files inside '
                                       'the given directory must have the form'
                                       ' $lang$.magnitude.')
