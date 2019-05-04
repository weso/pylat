from ..exceptions import InvalidArgumentError
from sklearn.metrics import confusion_matrix

import numpy as np

import math


# constant values used in the wilson score interval
confidence_interval_to_const = {
    0.90: 1.64,
    0.95: 1.96,
    0.98: 2.33,
    0.99: 2.58
}


def wilson_score_interval(error, n, confidence):
    try:
        const = confidence_interval_to_const[confidence]
    except KeyError:
        raise InvalidArgumentError('confidence', 'Invalid confidence value. '
                                   'Valid values are: {}'.format(
                                    confidence_interval_to_const.keys()))
    return const * math.sqrt(error * (1-error) / n)


def mcnemar_test(y_pred1, y_pred2, y_true):
    classif1_correct = [1 if y_pred1[i] == y_true[i] else 0
                        for i, _ in enumerate(y_pred1)]
    classif2_correct = [1 if y_pred2[i] == y_true[i] else 0
                        for i, _ in enumerate(y_pred2)]
    table = _build_contingency_table(classif1_correct, classif2_correct)
    return math.pow(table[0, 1] - table[1, 0], 2) / (table[0, 1] - table[1, 0])


def stuart_maxwell_test(y_pred1, y_pred2):
    conf_matrix = confusion_matrix(y_pred1, y_pred2)
    n = np.sum(conf_matrix, axis=1)
    n_prime = np.sum(conf_matrix, axis=0)
    _check_valid(conf_matrix, n, n_prime)
    k = conf_matrix.shape[0]
    d = [n[i] - n_prime[i] for i in range(k)]
    S = something really strange
    return np.transpose(d) * np.linalg.inv(S) * d

def _check_valid(conf_matrix, n, n_prime):
    # check that there is not agreement
    pass

def _build_contingency_table(clf1, clf2):
    table = np.zeros(shape=(2, 2), dtype=np.int)
    # clf1 and clf2 correct
    table[0, 0] = np.sum(clf1 == 1) + np.sum(clf2 == 1)
    # clf1 correct, clf2 incorrect
    table[0, 1] = np.sum(clf1 == 1) + np.sum(clf2 == 0)
    # clf1 incorrect, clf2 correct
    table[1, 0] = np.sum(clf1 == 0) + np.sum(clf2 == 1)
    # clf1 and clf2 incorrect
    table[1, 1] = np.sum(clf1 == 0) + np.sum(clf2 == 0)
    return table
