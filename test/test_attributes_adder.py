from pylat.exceptions import InvalidArgumentError
from pylat.wrapper.transformer import AttributesAdder

import numpy as np
import pytest


def identity_transformer(X):
    """Returns same input"""
    return X


def double_transformer(X):
    """Returns double of each element"""
    return np.asarray(X) * 2


class TestRNNWrapper:
    def test_empty_transformers(self):
        adder = AttributesAdder(transformers=[])
        output = adder.transform([[1, 2, 3], [4, 5, 6]])
        assert output.size == 0

    def test_invalid_transformers(self):
        with pytest.raises(InvalidArgumentError):
            AttributesAdder(transformers="abcde")

        with pytest.raises(InvalidArgumentError):
            AttributesAdder(transformers=[2, 3, 4])

    def test_valid_transformers(self):
        transformers = [identity_transformer, double_transformer]
        adder = AttributesAdder(transformers)
        adder.fit([])  # optional step
        output = adder.transform([[1], [4]])
        assert np.allclose(output, [[[1, 2],
                                     [4, 8]]])
