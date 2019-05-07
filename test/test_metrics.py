from pylat.evaluation.metrics import false_discovery_rate, false_negative_rate,\
    false_positive_rate, negative_predicted_value, positive_predicted_value
from pylat.exceptions import InvalidArgumentError

import pytest
import unittest


class TestMetricsMethods(unittest.TestCase):

    def setUp(self):
        self.y_pred = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        self.y_true = [2, 1, 3, 1, 2, 3, 1, 2, 1]

    def test_metrics_valid(self):
        assert pytest.approx(false_discovery_rate(self.y_true, self.y_pred),
                             1e-3) == 0.3333
        assert pytest.approx(false_negative_rate(self.y_true, self.y_pred),
                             1e-3) == 0.2777
        assert pytest.approx(false_positive_rate(self.y_true, self.y_pred),
                             1e-3) == 0.1698
        assert pytest.approx(negative_predicted_value(self.y_true, self.y_pred),
                             1e-3) == 0.8333
        assert pytest.approx(positive_predicted_value(self.y_true, self.y_pred),
                             1e-3) == 0.6666

    def test_metrics_invalid(self):
        y_true = [1, 2, 3, 4, 1]  # different length than y_pred
        with pytest.raises(InvalidArgumentError):
            false_discovery_rate(y_true, self.y_pred)
