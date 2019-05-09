from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from pylat.wrapper.transformer import TextPreprocessor
from pylat.exceptions import InvalidArgumentError

import logging
import pytest
import unittest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pipe_sample(doc):
    if len(doc) > 10:
        logger.info('This is a pretty long text!')
    else:
        logger.info('This is a pretty short text!')
    return doc


def do_nothing(x):
    return x


class TestTextPreprocessor(unittest.TestCase):

    def setUp(self):
        self.sample_en = [
            u"Hello, this is a simple text case :)",
            u"This is the 2nd sample",
            u"I would like to present you my brother-in-law",
            u"This is pretty sad",
            u"I need to make multiple samples",
            u"For the scikit learn grid search",
            u"I'm tired :("
        ]
        self.labels_en = [
            0,
            1,
            0,
            1,
            1,
            0,
            1
        ]

    def test_invalid_params(self):
        invalid_stop_word_type = TextPreprocessor(remove_stop_words=2)
        with pytest.raises(InvalidArgumentError):
            invalid_stop_word_type.fit(self.sample_en)

        invalid_lemma_type = TextPreprocessor(lemmatize=3)
        with pytest.raises(InvalidArgumentError):
            invalid_lemma_type.fit(self.sample_en)

        invalid_add_pipe_type = TextPreprocessor(additional_pipes='hi')
        with pytest.raises(InvalidArgumentError):
            invalid_add_pipe_type.fit(self.sample_en)

        invalid_model_id = TextPreprocessor(spacy_model_id='invented')
        with pytest.raises(InvalidArgumentError):
            invalid_model_id.fit(self.sample_en)

    def test_valid_params(self):
        text_processor = TextPreprocessor(spacy_model_id='es', remove_stop_words=True,
                                          lemmatize=True, additional_pipes=[pipe_sample])
        text_processor.fit(self.sample_en)
        transformed_text = text_processor.transform(self.sample_en)
        self.assertEqual(len(self.sample_en), len(transformed_text))
        for doc in transformed_text:
            self.assertTrue(len(doc) != 0)

    def test_sklearn_pipeline(self):
        text_preprocessor = TextPreprocessor(spacy_model_id='en', lemmatize=True)
        tf_idf_vectorizer = TfidfVectorizer(input='content', tokenizer=lambda x: x,
                                            preprocessor=None, lowercase=False)
        bayes_clf = MultinomialNB()

        pipe = Pipeline(steps=[
            ('preprocessing', text_preprocessor),
            ('tf-idf', tf_idf_vectorizer),
            ('clf', bayes_clf)
        ])
        pipe.fit(self.sample_en, self.labels_en)

    def test_search_cv(self):
        pipe = Pipeline(steps=[
            ('txt_prep', TextPreprocessor()),
            ('tf-idf', TfidfVectorizer(input='content', tokenizer=do_nothing,
                                       preprocessor=None, lowercase=False)),
            ('clf', MultinomialNB())
        ])

        param_grid = {
            'txt_prep__spacy_model_id': ['en', 'es'],
            'txt_prep__lemmatize': [True, False],
            'txt_prep__remove_stop_words': [True, False],
            'txt_prep__additional_pipes': [None, [pipe_sample]]
        }

        grid_search = GridSearchCV(pipe, param_grid, n_jobs=-1, iid=False, cv=2)
        grid_search.fit(self.sample_en, self.labels_en)

    def test_transform_before_fit(self):
        preprocessor = TextPreprocessor()
        with pytest.raises(NotFittedError):
            preprocessor.transform(self.sample_en)


if __name__ == '__main__':
    unittest.main()
