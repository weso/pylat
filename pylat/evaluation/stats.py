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
    """ Calculates the Wilson Score interval of a model with a given confidence.

    Parameters
    ----------
    error : float
        Error score of the classifier with the sample data. It must be a float
        between 0 and 1.
    n : int
        Number of samples in the data.
    confidence : float
        Value of confidence to be used when calculating the interval. Valid
        values are 0.90, 0.95, 0.98 and 0.99.

    Returns
    -------
    float
        Float that

    Examples
    --------
    >>> from pylat.evaluation.stats import wilson_score_interval
    >>> n = 200  # 200 samples
    >>> error = 0.15  # model misclassified 15% of the samples
    >>> print(wilson_score_interval(error, n, 0.90))  # 90% confidence

    >>> print(wilson_score_interval(error, n, 0.95))  # 95% confidence

    >>> print(wilson_score_interval(error, n, 0.99))  # 99% confidence

    >>> print(wilson_score_interval(error, n, 0.79))  # invalid confidence

    """
    try:
        const = confidence_interval_to_const[confidence]
    except KeyError:
        raise InvalidArgumentError('confidence', 'Invalid confidence value. '
                                   'Valid values are: {}'.format(
                                    confidence_interval_to_const.keys()))
    return const * math.sqrt(error * (1-error) / n)


def mcnemar_test(y_pred1, y_pred2, y_true):
    """Performs a McNemar test of two predictions compared to the true values.

    This method can be used to check if there is a significant difference
    between the predictions made by two different classifiers.

    Notes
    -----
    This method should only be applied to compare the outputs of two binary
    classifiers. In order to compare multiclass classifiers, please see
    :func:`~stuart_maxwell_test`.

    Parameters
    ----------
    y_pred1 : iterable
        Iterable containing every prediction made by the first classifier.
    y_pred2 : iterable
        Iterable containing every prediction made by the second classifier.
    y_true : iterable
        Iterable contining the true predictions.

    Returns
    -------
    float
        Float bounded between 0 and 1, containing the result of applying the
        McNemar test.
    """
    classif1_correct = [1 if y_pred1[i] == y_true[i] else 0
                        for i, _ in enumerate(y_pred1)]
    classif2_correct = [1 if y_pred2[i] == y_true[i] else 0
                        for i, _ in enumerate(y_pred2)]
    table = _build_contingency_table(classif1_correct, classif2_correct)
    return math.pow(table[0, 1] - table[1, 0], 2) / (table[0, 1] - table[1, 0])


def stuart_maxwell_test(y_pred1, y_pred2):
    """Perform a Stuart-Maxwell test to compare the results of two classifiers.

    Notes
    -----
    This method is equivalent to calling :func:`~mcnemar_test` if called with the
    predictions of two binary classifiers.

    If the two predictions agree on more than 1 label (they produce the same
    outpus for that label), this method will return an error since the
    Stuart-Maxwell test can't be calculated in that case.

    Parameters
    ----------
    y_pred1: iterable
        Iterable containing every prediction made by the first classifier.
    y_pred2
        Iterable containing every prediction made by the second classifier.

    Returns
    -------
    float
        Float bounded between 0 and 1 containing the result of the
    """
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
