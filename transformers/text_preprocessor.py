from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from ..exceptions import InvalidArgumentError

import numpy as np
import spacy

import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Basic text preprocessing class.

    This class provides several options for the preprocessing and
    cleaning of textual data, and it serves as a good first step before
    training models that work with this kind of data.

    It implements the BaseEstimator and TransformerMixin classes from
    sklearn, so it can be used in its pipelines and all of its hyperparameter
    search classes.

    Parameters
    ----------
    spacy_model_id : str, optional (default='en')
        Id of the Spacy model that will be used to perform the basic
        natural language operations (parser, tagger, ner...). The model
        must have been imported first (see https://spacy.io/usage/models).

    additional_pipes : iterable or None, optional (default=None)
        Iterable of callables that will be appended to the original pipeline
        of the Spacy model. This callables receive a Doc object as input,
        and return another Doc object.

    remove_stop_words : boolean, optional (default=False)
        Whether to remove stop_words from the input text or not.

    lemmatize : boolean, optional (default=False)
        Whether to return a lemmatized form of each token from the input text or not.

    Attributes
    ----------
    nlp : spacy.Language
        Loaded language model matching the received model id.

    custom_nlp : spacy.Language
        Final language model used for preprocessing, which includes
        the user defined pipes added in the 'additional_pipes' param.

    model_caches : dict,
        Dictionary that keeps a cache of all the loaded nlp models. Its keys
        are the string model ids, and its values are their respective nlp models.

    Examples
    --------
    >>> from src.text_preprocessor import TextPreprocessor
    >>> X = ['Hi, how are you doing?']
    >>> base_preprocessor = TextPreprocessor()
    >>> print(base_preprocessor.fit_transform(X))
    [['Hi' ',' 'how' 'are' 'you' 'doing' '?']]
    >>> lemmatizer = TextPreprocessor(lemmatize=True)
    >>> print(lemmatizer.fit_transform(X))
    [['hi' ',' 'how' 'be' '-pron-' 'do' '?']]
    >>> preprocessor = TextPreprocessor(remove_stop_words=True)
    >>> print(preprocessor.fit_transform(X))
    [['Hi']]
    """

    model_caches = {}

    def __init__(self, spacy_model_id='en', additional_pipes=None,
                 remove_stop_words=False, lemmatize=False,
                 disable=()):
        self.spacy_model_id = spacy_model_id
        self.remove_stop_words = remove_stop_words
        self.lemmatize = lemmatize

        if additional_pipes is None:
            additional_pipes = []
        self.additional_pipes = additional_pipes
        self.disable = disable
        self.custom_nlp = None

    @property
    def nlp(self):
        """Spacy language model loaded given the id passed in the constructor.

        A dict of cached models is updated in order to avoid the expensive operation
        of loading a Language object.

        :return: spacy.Language object
            Spacy language model.
        """
        if self.spacy_model_id not in self.model_caches:
            logger.info('Model {} was not found in cache'.format(self.spacy_model_id))
            logger.info('Loading model {}...'.format(self.spacy_model_id))
            self.model_caches[self.spacy_model_id] = spacy.load(self.spacy_model_id, disable=self.disable)

        return self.model_caches[self.spacy_model_id]

    def fit(self, x, y=None):
        """Fit to data.

        This method must be called before transform. It doesn't take
        into account any of the data received as parameters, but it
        validates the params passed in the constructor, following
        sklearn conventions.

        :param x: any
            Not taken into account.
        :param y: any (default=None)
            Not taken into account.
        :return: self
        """
        self._assert_valid_types()

        # add user defined pipes to base nlp object
        self.custom_nlp = copy.deepcopy(self.nlp)
        for pipe in self.additional_pipes:
            logger.debug('Adding pipe "{}" to nlp pipeline.'.format(pipe))
            self.custom_nlp.add_pipe(pipe)

        return self

    def transform(self, x):
        """Transforms the input text into preprocessed data.

        :param x: array of strings of any shape (n)
        :return: numpy array of shape of shape [n, ]
            Each row in the output array corresponds to each document in the
            input array, and it contains a list of preprocessed tokens.
        """
        if self.custom_nlp is None:
            raise NotFittedError

        ret = []
        for sentence in x:
            doc = self.custom_nlp(sentence)
            ret.append(self._process(doc))

        return np.asarray(ret)

    def _process(self, doc):
        tokens = np.asarray(doc)
        if self.remove_stop_words:
            tokens = [tok for tok in tokens if not tok.is_stop and not tok.is_punct]

        if self.lemmatize:
            tokens = [tok.lemma_.lower().strip() for tok in tokens]
        else:
            tokens = [tok.text for tok in tokens]
        return tokens

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter == 'additional_pipes' and value is None:
                setattr(self, parameter, [])
                continue

            setattr(self, parameter, value)
        return self

    def _assert_valid_types(self):
        if not isinstance(self.remove_stop_words, bool):
            raise InvalidArgumentError('remove_stop_words', 'Remove stop words option must be True or False.')

        if not isinstance(self.lemmatize, bool):
            raise InvalidArgumentError('lemmatize', 'Lemmatize option must be True or False.')

        if not hasattr(self.additional_pipes, '__iter__'):
            logger.info('Additional pipes is: {}'.format(self.additional_pipes))
            raise InvalidArgumentError('additional_pipes', 'Additional pipes must be an iterable of callables')

        for pipe in self.additional_pipes:
            if not hasattr(pipe, '__call__'):
                raise InvalidArgumentError('additional_pipes', "All additional pipes must be callables")
