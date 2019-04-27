from sklearn.base import BaseEstimator, TransformerMixin

import abc


class BaseTransformer(abc.ABC, BaseEstimator, TransformerMixin):
    @abc.abstractmethod
    def fit(self, x, y=None):
        pass

    @abc.abstractmethod
    def transform(self, x):
        pass


class BasePredictor(abc.ABC, BaseEstimator):
    @abc.abstractmethod
    def predict(self, x, y):
        pass


class DataPipeline(BaseTransformer, BasePredictor):
    def __init__(self, transformers, predictor=None):
        self.transformers = transformers
        self.predictor = predictor
        self._assert_valid_params()

    def fit(self, x, y=None):
        for transformer in self.transformers:
            transformer.fit(x, y)

    def transform(self, x):
        transformed_x = x
        del x
        for transformer in self.transformers:
            transformed_x = transformer.transform(transformed_x)
        return transformed_x

    def predict(self, x, y):
        if self.predictor is None:
            raise Exception("The pipeline must have a predictor in order to call its predict method.")
        return self.predictor.predict(x, y)

    def _assert_valid_params(self):
        for transformer in self.transformers:
            assert(hasattr(transformer, "fit"))
            assert(hasattr(transformer, "transform"))

        assert(hasattr(self.predictor, "predict"))
