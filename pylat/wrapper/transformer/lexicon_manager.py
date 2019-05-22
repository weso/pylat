from pylat.exceptions import InvalidArgumentError

import csv


class LexiconManager():
    """ This class produces new features from a text using a Lexicon.

    The LexiconManager class uses the NRC Valence, Arousal and Dominance
    lexicon and provides function the return a score for each of the three
    dimensions given a piece of text. This class can be used for NLP
    applications in order to add new features to a classifier.

    Parameters
    ----------
    lexicon_path : str
        Path to the lexicon text file.

    Examples
    --------
    >>> import os
    >>> from pylat.wrapper.transformer.lexicon_manager import LexiconManager
    >>> tokens = ['I', 'need', 'more', 'alcohol', 'now']
    >>> lexicon_path = os.path.join('test', 'data', 'test_lexicon.txt')
    >>> lexicon = LexiconManager(lexicon_path)
    >>> round(lexicon.valence(tokens), 3)
        0.178
    >>> round(lexicon.arousal(tokens), 3)
        0.188
    >>> round(lexicon.dominance(tokens), 3)
        0.304
    """
    def __init__(self, lexicon_path):
        self._lexicon = dict()
        with open(lexicon_path, newline='') as csvfile:
            lexicon_reader = csv.DictReader(csvfile, delimiter='\t')
            for row in lexicon_reader:
                name = row['Word']
                valence = row['Valence']
                arousal = row['Arousal']
                dominance = row['Dominance']
                self._lexicon[name] = LexiconRow(valence, arousal, dominance)

    def valence(self, text):
        """Returns the valence score for the given text.

        The valence score is a float between 0 and 1 that represents
        the pleasure--displeasure dimension of the text. A value near
        0 means that the text represents displeasure, while a value near
        1 means that the text represents pleasure.

        Parameters
        ----------
        text : :obj:`list` of str
            Iterable of tokens present in the text.

        Returns
        -------
        return : float
            Normalized number with the valence score.
        """
        return self._obtain_lexicon_info('valence', text)

    def arousal(self, text):
        """Returns the arousal score for the given text.

        The arousal score is a float between 0 and 1 that represents
        the active--passive dimension of the text. A value

        Parameters
        ----------
        text : :obj:`list` of str
            String representation of the text.

        Returns
        -------
        return : float
            Normalized number with the arousal score.
        """
        return self._obtain_lexicon_info('arousal', text)

    def dominance(self, text):
        """Returns the dominance score for the given text.

        Parameters
        ----------
        text : :obj:`list` of str
            String representation of the text.

        Returns
        -------
        return : float
            Normalized number with the dominance score.
        """
        return self._obtain_lexicon_info('dominance', text)

    def _obtain_lexicon_info(self, variable, text):
        """ Calculates a specific dimension from the lexicon.

        Parameters
        ----------
        variable : str
            Must be one of 'valence', 'arousal' and 'dominance'.
        text : :obj:`list` of str
            List of tokens which are present in the text.

        Returns
        -------
        return : float
            Normalized number with the specified score.
        """
        if not hasattr(text, '__iter__') or isinstance(text, str):
            raise InvalidArgumentError('text', 'Text must be an iterable of tokens')
        result = 0
        words_used = 0
        for token in text:
            lexicon_data = self._lexicon.get(token)
            if lexicon_data is not None:
                words_used += 1
                result += float(getattr(lexicon_data, variable))
        return 0.5 if words_used == 0 else result / words_used


class LexiconRow:
    """Data class that represents every row present in the lexicon.

    This class is used to store every possible value present in the NRC
    valence, arousal and dominance lexicon.

    Parameters
    ----------
    valence : float
        Value of the valence score of the word.
    arousal : float
        Value of the arousal score of the word.
    dominance : float
        Value of the dominance score of the word.
    """
    def __init__(self, valence, arousal, dominance):
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
