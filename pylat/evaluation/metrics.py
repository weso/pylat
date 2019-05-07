from pylat.exceptions import InvalidArgumentError

from sklearn.metrics import confusion_matrix

import numpy as np


def false_discovery_rate(y_true, y_pred):
    """Calculate the false discovery rate (FDR) of a classifier.

    The false discovery rate is a value that represents the ratio
    of type I errors made by a classifier. It is obtained by dividing
    the number of false positives by the classifier over all the number
    of positives predicted.

    Parameters
    ----------
    y_true : iterable
        Correct set of labels of the sample.
    y_pred : iterable
        Set of labels predicted by the classifier.

    Returns
    -------
    float
        Float value between 0 and 1 representing the false discovery rate of
        the classifier.

    Examples
    --------
    >>> from pylat.evaluation.metrics import false_discovery_rate
    >>> y_pred = [1, 0, 0, 1, 0, 2, 0, 0, 2, 2, 0]
    >>> y_true = [1, 1, 0, 0, 1, 1 ,2, 1, 2, 2, 0]
    >>> print(round(false_discovery_rate(y_true, y_pred), 3))
    0.5
    """
    fp, fn, tp, tn = _compute_values_from(y_true, y_pred)
    return np.mean(fp / (tp + fp))


def false_negative_rate(y_true, y_pred):
    """Calculate the false negative rate (FNR) of a classifier.

    The false negative rate is a value that represents the proportion of
    positive values in the data that are predicted as negative by the
    classifier.

    Parameters
    ----------
    y_true : iterable
        Correct set of labels of the sample.
    y_pred : iterable
        Set of labels predicted by the classifier.

    Returns
    -------
    float
        Float value between 0 and 1 representing the false negative rate of
        the classifier.

    Examples
    --------
    >>> from pylat.evaluation.metrics import false_negative_rate
    >>> y_pred = [1, 0, 0, 1, 0, 2, 0, 0, 2, 2, 0]
    >>> y_true = [1, 1, 0, 0, 1, 1 ,2, 1, 2, 2, 0]
    >>> print(round(false_negative_rate(y_true, y_pred), 3))
    0.489
    """
    fp, fn, tp, tn = _compute_values_from(y_true, y_pred)
    return np.mean(fn / (tp + fn))


def false_positive_rate(y_true, y_pred):
    """Calculate the false positive rate (FPR) of a classifier.

    The false discovery rate is a value that represents the proportion of
    negative samples in the data that are returned as positive by the
    classifier.

    Parameters
    ----------
    y_true : iterable
        Correct set of labels of the sample.
    y_pred : iterable
        Set of labels predicted by the classifier.
    Returns
    -------
    float
        Float value between 0 and 1 representing the false positive rate of
        the classifier.

    Examples
    --------
    >>> from pylat.evaluation.metrics import false_positive_rate
    >>> y_pred = [1, 0, 0, 1, 0, 2, 0, 0, 2, 2, 0]
    >>> y_true = [1, 1, 0, 0, 1, 1 ,2, 1, 2, 2, 0]
    >>> print(round(false_positive_rate(y_true, y_pred), 3))
    0.264
    """
    fp, fn, tp, tn = _compute_values_from(y_true, y_pred)
    return np.mean(fp / (fp + tn))


def negative_predicted_value(y_true, y_pred):
    """Calculate the negative predicted value (NPV) of a classifier.

    The negative predicted value represents the proportion of
    true negatives returned by the classifier with respect to all the negative
    samples in the data.

    Parameters
    ----------
    y_true : iterable
        Correct set of labels of the sample.
    y_pred : iterable
        Set of labels predicted by the classifier.

    Returns
    -------
    float
        Float value between 0 and 1 representing the negative predicted value
        of the classifier.

    Examples
    --------
    >>> from pylat.evaluation.metrics import negative_predicted_value
    >>> y_pred = [1, 0, 0, 1, 0, 2, 0, 0, 2, 2, 0]
    >>> y_true = [1, 1, 0, 0, 1, 1 ,2, 1, 2, 2, 0]
    >>> print(round(negative_predicted_value(y_true, y_pred), 3))
    0.744
    """
    fp, fn, tp, tn = _compute_values_from(y_true, y_pred)
    return np.mean(tn / (tn + fn))


def positive_predicted_value(y_true, y_pred):
    """Calculate the positive predicted value (PPV) of a classifier.

    The positive predicted value represents the proportion of true
    positives returned by the classifier with respect to all the positive
    samples in the data.

    Parameters
    ----------
    y_true : iterable
        Correct set of labels of the sample.
    y_pred : iterable
        Set of labels predicted by the classifier.

    Returns
    -------
    float
        Float value between 0 and 1 representing the positive predicted value
        of the classifier.

    Examples
    --------
    >>> from pylat.evaluation.metrics import positive_predicted_value
    >>> y_pred = [1, 0, 0, 1, 0, 2, 0, 0, 2, 2, 0]
    >>> y_true = [1, 1, 0, 0, 1, 1 ,2, 1, 2, 2, 0]
    >>> print(round(positive_predicted_value(y_true, y_pred), 3))
    0.5
    """
    fp, fn, tp, tn = _compute_values_from(y_true, y_pred)
    return np.mean(tp / (tp + fp))


def _compute_values_from(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise InvalidArgumentError('y_true', 'Both predictions must contain the'
                                             'same number of elements.')

    conf_mtx = confusion_matrix(y_true, y_pred)
    tp = np.diag(conf_mtx)
    fp = conf_mtx.sum(axis=0) - tp
    fn = conf_mtx.sum(axis=1) - tp
    tn = conf_mtx.sum() - (tp + fp + fn)
    return fp, fn, tp, tn
