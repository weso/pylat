from pylat.exceptions import InvalidArgumentError
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Returns a subset of a pandas DataFrame as a numpy array.

    This transformer is meant to be used as the first transformer in a
    scikit-learn pipeline. It will receive a pandas DataFrame as input,
    and the names of the columns that you want to extract from the DataFrame
    for later use in the pipeline by other transformers.

    Parameters
    ----------
    column_names : :obj:`list` of str or str
        List containing the names of the columns that will be extracted
        from the input DataFrame. If you only want to extract one column,
        a single string can be passed instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.pipeline import Pipeline
    >>> from pylat.wrapper.transformer.dataframe_selector import DataFrameSelector
    >>> df = pd.DataFrame(data={
    ...    'text': ['Sample text', 'another one', 'more text data', 'last one one'],
    ...    'labels': [0, 0, 1, 0],
    ...    'some_column': [0.01, 0.2, 0.1, 0.7]
    ... })
    >>> text_pipeline = Pipeline(steps=[
    ...     ('selector', DataFrameSelector('text')),
    ...     ('tf_idf', CountVectorizer())
    ... ])
    >>> print(text_pipeline.fit_transform(df).toarray())
    [[0 0 0 0 0 1 1]
     [1 0 0 0 1 0 0]
     [0 1 0 1 0 0 1]
     [0 0 1 0 2 0 0]]
    """

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, x, y=None, **fit_params):
        return self

    def transform(self, x):
        """Extract the given column_names from x.

        Parameters
        ----------
        x : pandas DataFrame
            Pandas DataFrame from which the columns will be extracted.

        Returns
        -------
        return : numpy array
            Numpy array containing the extracted columns.

        Raises
        ------
        InvalidArgumentError
            If x is not a pandas DataFrame.

        KeyError
            If the column names passed in the constructor of the
            selector don't exist in the given DataFrame.
        """
        if not isinstance(x, pd.DataFrame):
            raise InvalidArgumentError('x', 'Transform must receive a pandas '
                                            'Dataframe as argument.')

        return x[self.column_names].values
