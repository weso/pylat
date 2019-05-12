from pylat.wrapper.transformer import DataFrameSelector
from pylat.exceptions import InvalidArgumentError

import numpy as np
import pandas as pd
import pytest


class TestDataFrameSelector():
    def test_valid_df(self):
        df = pd.DataFrame(data={
            'important': [50, 2, 7],
            'not_important': [1, 3, 5]
        })
        selector = DataFrameSelector(column_names='important')
        res = selector.fit_transform(df)
        assert np.array_equal(res, [50, 2, 7])

    def test_invalid_df(self):
        df = 'Hi :)'
        selector = DataFrameSelector(column_names=['important'])
        with pytest.raises(InvalidArgumentError):
            selector.fit_transform(df)

    def test_valid_selection(self):
        df = pd.DataFrame(data={
            'a_column': [1, 2, 4, 1],
            'labels': ['Y', 'N', 'Y', 'N/A'],
            'text': ['Helloooo', 'This is a sample', 'Nice', 'Can you come here?'],
            'another_column': [0.45, 0.1213, 0.134, 0.0001]
        })
        selector = DataFrameSelector(column_names=['labels', 'text'])
        res = selector.fit_transform(df)
        assert np.array_equal(res, [
            ['Y', 'Helloooo'],
            ['N', 'This is a sample'],
            ['Y', 'Nice'],
            ['N/A', 'Can you come here?']
        ])

    def test_invalid_selection(self):
        df = pd.DataFrame(data={
            'a_column': [1, 2, 4, 1]
        })
        selector = DataFrameSelector(column_names=[2, 'muahahahah'])
        with pytest.raises(KeyError):
            selector.fit_transform(df)
