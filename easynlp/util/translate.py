# -*- coding: utf-8 -*-

import logging, json, requests, uuid

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TextTranslator():
    def __init__(self, translator_key):
        self.base_url = 'https://api.cognitive.microsofttranslator.com'
        self.path = '/translate?api-version=3.0'
        self.headers = {
            'Ocp-Apim-Subscription-Key': translator_key,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }

    def translate(self, text, src, to):
        params = '&from={}&to={}'.format(src, to)
        constructed_url = self.base_url + self.path + params
        body = [{
            'text' : text[:5000] # limit maximum number of characters to 5000
        }]
        request = requests.post(constructed_url, json=body, headers=self.headers)
        output = request.json()
        logger.info('Text: {} - Output: {}'.format(text, output))
        return output
