from pylat.evaluation.stats import wilson_score_interval, mcnemar_test, \
    stuart_maxwell_test
from pylat.exceptions import InvalidArgumentError

import pytest


class TestStatsMethods:
    def test_wilson_valid(self):
        error = 0.2
        n_samples = 300
        assert pytest.approx(wilson_score_interval(error, n_samples, 90),
                             1e-3) == 0.03787
        assert pytest.approx(wilson_score_interval(error, n_samples, 95),
                             1e-3) == 0.04526
        assert pytest.approx(wilson_score_interval(error, n_samples, 98),
                             1e-3) == 0.05381
        assert pytest.approx(wilson_score_interval(error, n_samples, 99),
                             1e-3) == 0.05958

    def test_wilson_invalid_error(self):
        with pytest.raises(InvalidArgumentError):
            wilson_score_interval(-0.01, 300, 90)
        with pytest.raises(InvalidArgumentError):
            wilson_score_interval(1.01, 300, 90)

    def test_wilson_invalid_samples(self):
        with pytest.raises(InvalidArgumentError):
            wilson_score_interval(0.2, -9, 90)

    def test_wilson_invalid_confidence(self):
        with pytest.raises(InvalidArgumentError):
            wilson_score_interval(0.2, 100, 92)

    def test_mcnemar_one_label(self):
        y_pred1 = [1, 1, 1, 1, 1]
        y_pred2 = [1, 1, 1, 1, 1]
        y_true = [1, 1, 1, 1, 1]
        with pytest.raises(InvalidArgumentError):
            mcnemar_test(y_pred1, y_pred2, y_true)

    def test_mcnemar_two_labels(self):
        y_pred1 = ['a', 'a', 'a', 'b', 'b']
        y_pred2 = ['b', 'a', 'b', 'a', 'a']
        y_true = ['b', 'a', 'b', 'a', 'b']
        assert mcnemar_test(y_pred1, y_pred2, y_true) == 1

    def test_mcnemar_multiple_labels(self):
        y_pred1 = ['a', 'a', 'b', 'c']
        y_pred2 = ['b', 'c', 'b', 'a']
        y_true = ['c', 'a', 'c', 'b']
        with pytest.raises(InvalidArgumentError):
            mcnemar_test(y_pred1, y_pred2, y_true)

    def test_mcnemar_invalid(self):
        y_pred1 = ['a', 'a']
        y_pred2 = ['a', 'a', 'c']
        y_true = ['a']
        with pytest.raises(InvalidArgumentError):
            mcnemar_test(y_pred1, y_pred2, y_true)

    def test_stuart_no_agreement(self):
        y_pred1 = ['a', 'b', 'c', 'b', 'a', 'c']
        y_pred2 = ['b', 'a', 'b', 'b', 'a', 'c']
        assert pytest.approx(stuart_maxwell_test(y_pred1, y_pred2),
                             1e-4) == 0.6065

    def test_stuart_one_agreement(self):
        y_pred1 = ['a', 'b', 'c', 'b', 'a', 'c']
        y_pred2 = ['b', 'b', 'c', 'c', 'a', 'c']
        assert pytest.approx(stuart_maxwell_test(y_pred1, y_pred2),
                             1e-4) == 0.3679

    def test_stuart_multiple_agreements(self):
        y_pred1 = ['a', 'b', 'c', 'b', 'a', 'd']
        y_pred2 = ['b', 'a', 'c', 'b', 'a', 'd']
        with pytest.raises(InvalidArgumentError):
            stuart_maxwell_test(y_pred1, y_pred2)

    def test_stuart_invalid(self):
        y_pred1 = ['a', 'b', 'c', 'd']
        y_pred2 = ['a', 'b', 'c']
        with pytest.raises(InvalidArgumentError):
            stuart_maxwell_test(y_pred1, y_pred2)
