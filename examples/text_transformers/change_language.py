""" This script illustrates how to preprocess texts
written in different languages.
"""
import sys
from pylat.wrapper.transformer import TextPreprocessor


def main():
    # Our first text is going to be written in english
    text_en = u"We are going to test my experimental new functionality."

    # By default, the TextPreprocessor excepts texts to be written
    # in English, here we explicitly tell it so by passing the parameter
    # 'en' in the constructor.
    preprocessor_en = TextPreprocessor('en')
    print(preprocessor_en.fit_transform([text_en]))

    # The following text is written in english
    text_es = u"Este texto está escrito en español."

    # In this case, we pass the parameter 'es' to indicate
    # that the text is written in spanish.
    preprocessor_es = TextPreprocessor('es')
    print(preprocessor_es.fit_transform([text_es]))



if __name__ == '__main__':
    sys.exit(main())
