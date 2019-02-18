import pandas as pd
import xmltodict

import csv
import logging
import os

from .data_sample import DataSample


class DataParser:
    def __init__(self, data_path, labels_file_name, posts_dir_name):
        self.labels_file = os.path.join(data_path, labels_file_name)
        self.posts_dir = os.path.join(data_path, posts_dir_name)
        self._labeled_dataframe = None
        self._complete_dataframe = None

    @property
    def labeled_dataframe(self):
        if self._labeled_dataframe is not None:
            return self._labeled_dataframe
        else:
            self._labeled_dataframe = self._compute_dataframe(labeled_only=True)
            return self.labeled_dataframe

    @property
    def complete_dataframe(self):
        if self._complete_dataframe is not None:
            return self._complete_dataframe
        else:
            self._complete_dataframe = self._compute_dataframe()
            return self._complete_dataframe

    def _compute_dataframe(self, labeled_only=False):
        logging.info('Parsing labels...')
        self.labels = self._parse_labels()
        logging.info('Parsing posts...')
        if labeled_only:
            posts = [self._parse_sample(post) for post in os.listdir(self.posts_dir)
                     if self._is_labeled(post)]
        else:
            posts = [self._parse_sample(post) for post in os.listdir(self.posts_dir)]
        logging.info('Creating dataframe...')
        df = self._construct_df_from(posts)
        return df

    def _is_labeled(self, post):
        for label in self.labels.keys():
            if '-{}.xml'.format(label) in post:
                return True
        return False

    def _parse_sample(self, xml_doc):
        with open(os.path.join(self.posts_dir, xml_doc), 'r', encoding='utf8') as f:
            xml_data = f.read()
        tree = xmltodict.parse(xml_data)
        message = tree['response']['message']
        msg_id = int(message['id']['#text'])
        board_id = int(message['board_id']['#text'])
        root_id = DataParser._extract_id_from(message['root']['@href'])
        kudos = int(message['kudos']['count']['#text'])
        post_time = message['post_time']['#text']
        edit_time = message['last_edit_time']['#text']
        msg_data = DataParser._get_text(message)
        thread_id = DataParser._extract_id_from(message['thread']['@href'])
        parent_msg_id = DataParser._extract_parent_id_from(message)
        views = message['views']['count']['#text']
        author_id = DataParser._extract_id_from(message['author']['@href'])
        label = self.labels.get(msg_id)
        return DataSample(msg_id, msg_data, board_id, root_id,
                          kudos, author_id, post_time, edit_time,
                          thread_id, parent_msg_id, views, label)

    def _parse_labels(self):
        with open(self.labels_file) as csv_file:
            reader = csv.reader(csv_file, delimiter='\t')
            logging.debug('Reader created')
            labels = {int(row[0]): row[1] for row in reader}
            logging.debug('Labels obtained')
        return labels

    def _construct_df_from(self, posts):
        df = pd.DataFrame.from_dict([post.to_df_data() for post in posts])
        return df

    @classmethod
    def _get_text(cls, message):
        try:
            return message['body']['#text']
        except KeyError:
            return ''

    @classmethod
    def _extract_id_from(cls, href):
        return href.split('/')[-1]

    @classmethod
    def _extract_parent_id_from(cls, message):
        try:
            parent_id = message['parent']['@href']
            return parent_id
        except KeyError:
            return None
