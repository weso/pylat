""" This script illustrates how to add custom functions to
the text preprocessing pipeline.
"""
import sys
from pylat.wrapper.transformer import TextPreprocessor


def my_new_function(doc):
    """ Custom function to add to the text preprocessing pipeline.

    This function just reads the length of the document and prints
    a message if the text is long and another one if its short.
    We could also modify the document object here. For more information
    about the attributes and functions provided by the object please
    visit https://spacy.io/api/doc.
    """
    if len(doc) >= 10:
        print('Doc "{}" is really long.'.format(doc))
    else:
        print('Doc "{}" is short.'.format(doc))
    return doc

def main():
    # Text creation
    texts = [
        u"We are going to test my experimental new functionality.",
        u"This is the 2nd sample.",
        u"And this is the final one"
    ]

    # adding our function to the pipeline
    # more functions could be added to the list.
    preprocessor = TextPreprocessor(additional_pipes=[my_new_function])

    # the transform method will call our custom function
    preprocessor.fit_transform(texts)


if __name__ == '__main__':
    sys.exit(main())
