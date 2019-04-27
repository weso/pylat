from pylat.wrapper.transformer.lexicon_manager import LexiconManager
from pylat.exceptions import InvalidArgumentError

import os
import pytest

lexicon_path = os.path.join('test', 'data', 'test_lexicon.txt')


class TestLexiconManager():
    def test_valid_text(self):
        """ Test with tokens that are contained in the lexicon.
        """
        tokens = ["this", "first", "test", "is", "mine"]
        lexicon = LexiconManager(lexicon_path)
        assert pytest.approx(lexicon.valence(tokens), 0.001) == 0.270
        assert pytest.approx(lexicon.arousal(tokens), 0.001) == 0.220
        assert pytest.approx(lexicon.dominance(tokens), 0.001) == 0.120

    def test_invalid_lexicon_path(self):
        """ Test passing an invalid lexicon path in the constructor.
        """
        with pytest.raises(FileNotFoundError):
            LexiconManager('invented')

    def test_invalid_text(self):
        """ Test with invalid token format.
        """
        text = "This is an untokenized text"
        lexicon = LexiconManager(lexicon_path)
        with pytest.raises(InvalidArgumentError):
            lexicon.valence(text)

    def test_text_without_info(self):
        """ Test where there is no token contained in the lexicon.
        """
        tokens = ['a', 'b', 'c']
        lexicon = LexiconManager(lexicon_path)
        assert pytest.approx(lexicon.valence(tokens), 0.1) == 0.5
        assert pytest.approx(lexicon.arousal(tokens), 0.1) == 0.5
        assert pytest.approx(lexicon.dominance(tokens), 0.1) == 0.5
