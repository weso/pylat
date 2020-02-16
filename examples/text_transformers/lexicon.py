""" This scripts shows how we can calculate numerical information
about the text using a VAD lexicon (see https://saifmohammad.com/WebPages/nrc-vad.html)
"""

import os
import sys

from pylat.wrapper.transformer import LexiconManager


def main():
    # We assume the text is already tokenized
    # this could be the output of our TextPreprocessor class
    tokens = ['I', 'need', 'more', 'alcohol', 'now']

    # Path to our lexicon file
    lexicon_path = os.path.join('data', 'test_lexicon.txt')
    lexicon = LexiconManager(lexicon_path)

    # use the lexicon to compute the VAD score for our tokens
    print('Arousal: {:.3f}\nDominance: {:.3f}\nValence: {:.3f}'.format(
        lexicon.arousal(tokens),
        lexicon.dominance(tokens),
        lexicon.valence(tokens)
    ))

if __name__ == '__main__':
    sys.exit(main())
