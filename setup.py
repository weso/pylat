from setuptools import setup

setup(
    name='pylat',
    version='0.1.0a0',
    packages=['pylat', 'pylat.data', 'pylat.util', 'pylat.wrapper',
              'pylat.wrapper.predictor', 'pylat.wrapper.transformer',
              'pylat.neuralnet', 'pylat.neuralnet.rnn'],
    url='https://github.com/alejgh/pylat',
    license='MIT',
    author='Alejandro González Hevia',
    author_email='alejandrgh11@gmail.com',
    description='A simple library with some common nlp operations',
    install_requires=[
        'gensim', 'scikit-learn', 'numpy', 'pandas',
        'spacy', 'tensorflow==1.13.1', 'xmltodict',
        'en_core_web_sm', 'es_core_news_sm'
    ],
    dependency_links=[
        'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz#egg=en_core_web_sm-2.1.0',
        'https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-2.1.0/es_core_news_sm-2.1.0.tar.gz#egg=es_core_news_sm-2.1.0'
    ]
)
