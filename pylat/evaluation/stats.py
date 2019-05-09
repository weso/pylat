from ..exceptions import InvalidArgumentError
from sklearn.metrics import confusion_matrix

from scipy import stats

import numpy as np

import math


# constant values used in the wilson score interval
confidence_interval_to_const = {
    90: 1.64,
    95: 1.96,
    98: 2.33,
    99: 2.58
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
    confidence : int
        Value of confidence to be used when calculating the interval. Valid
        values are 90, 95, 98 and 99.

    Returns
    -------
    float
        Float that

    Examples
    --------
    >>> from pylat.evaluation.stats import wilson_score_interval
    >>> n = 200  # 200 samples
    >>> error = 0.15  # model misclassified 15% of the samples
    >>> print(round(wilson_score_interval(error, n, 90), 3))  # 90% confidence
    0.041
    >>> print(round(wilson_score_interval(error, n, 95), 3))  # 95% confidence
    0.049
    >>> print(round(wilson_score_interval(error, n, 99), 3))  # 99% confidence
    0.065
    """
    if error < 0 or error > 1:
        raise InvalidArgumentError('error', 'Invalid error value. Error must '
                                            'be a value between 0 and 1.')
    elif n <= 0:
        raise InvalidArgumentError('n', 'Number of samples must be greater '
                                        'than 0.')

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
    if len(y_pred1) != len(y_pred2) or len(y_pred1) != len(y_true):
        raise InvalidArgumentError('y_pred', 'Predictions must have '
                                             'the same length.')
    elif len(np.unique(y_true)) != 2:
        raise InvalidArgumentError('num_labels', 'Number of labels must be '
                                                 'equal to 2. Use the stuart-'
                                                 'maxwell test instead.')

    classif1_correct = [1 if y_pred1[i] == y_true[i] else 0
                        for i, _ in enumerate(y_pred1)]
    classif2_correct = [1 if y_pred2[i] == y_true[i] else 0
                        for i, _ in enumerate(y_pred2)]
    table = _build_contingency_table(classif1_correct, classif2_correct)
    return math.pow(table[0, 1] - table[1, 0], 2) / (table[0, 1] + table[1, 0])


def stuart_maxwell_test(y_pred1, y_pred2):
    """Perform a Stuart-Maxwell test to compare the results of two classifiers.

    Notes
    -----
    This method is equivalent to calling :func:`~mcnemar_test` if called with
    the predictions of two binary classifiers.

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
    if len(y_pred1) != len(y_pred2):
        raise InvalidArgumentError('y_pred', 'Both predictions must have '
                                             'the same length.')
    conf_matrix = confusion_matrix(y_pred1, y_pred2)
    n = np.sum(conf_matrix, axis=1)
    n_prime = np.sum(conf_matrix, axis=0)
    k = conf_matrix.shape[0]
    conf_matrix, n, n_prime = _remove_agreements(conf_matrix, n, n_prime)
    d = [n[i] - n_prime[i] for i in range(k - 1)]
    s = _build_s_matrix(conf_matrix, n, n_prime, k)
    chi2 = np.dot(np.dot(d, np.linalg.inv(s)), np.transpose(d))
    return stats.distributions.chi2.sf(chi2, 2)


def _build_s_matrix(conf_matrix, n, n_prime, k):
    s = np.zeros(shape=(k-1, k-1))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if i == j:
                s[i, j] = n[i] + n_prime[i] - 2*conf_matrix[i, i]
            else:
                s[i, j] = -(conf_matrix[i, j] + conf_matrix[j, i])
    return s


def _remove_agreements(conf_matrix, n, n_prime):
    # check that there is not agreement
    agreement_rows = []
    for i, _ in enumerate(n):
        if conf_matrix[i][i] == n[i] and n[i] == n_prime[i]:
            agreement_rows.append(i)
    if len(agreement_rows) == 2:
        raise InvalidArgumentError('y_pred', "Stuart maxwell test can't be"
                                             "performed when the classifiers"
                                             "agree in more than one label.")
    elif len(agreement_rows) == 1:
        invalid_idx = agreement_rows[0]
        conf_matrix = np.delete(conf_matrix, invalid_idx, axis=0)
        conf_matrix = np.delete(conf_matrix, invalid_idx, axis=1)
        n = np.delete(n, invalid_idx, axis=0)
        n_prime = np.delete(n_prime, invalid_idx, axis=0)
    return conf_matrix, n, n_prime


def _build_contingency_table(clf1, clf2):
    table = np.zeros(shape=(2, 2), dtype=np.int)
    combined = [(i, j) for i, j in zip(clf1, clf2)]
    # clf1 and clf2 correct
    table[0, 0] = sum([1 if x == (1, 1) else 0 for x in combined])
    # clf1 correct, clf2 incorrect
    table[0, 1] = sum([1 if x == (1, 0) else 0 for x in combined])
    # clf1 incorrect, clf2 correct
    table[1, 0] = sum([1 if x == (0, 1) else 0 for x in combined])
    # clf1 and clf2 incorrect
    table[1, 1] = sum([1 if x == (0, 0) else 0 for x in combined])
    return table
