import os
import json
import logging
import numpy as np

from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BatchEncoding
)

from .args import BertCxConfig

logger = logging.getLogger(__name__)


class BertClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[str]] = None,
                 lbs: Optional[List[int]] = None,
                 encoded_texts: Optional[BatchEncoding] = BatchEncoding()
                 ):
        super().__init__()
        self._text = text
        self._lbs = lbs
        # splitted text so that every sentence is within maximum length when they are converted to BERT tokens
        self._encoded_texts = encoded_texts

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text_: List[str]):
        logger.warning("Setting text instances. Need to run `encode_text` or `encode_text_and_lbs` "
                       "to update encoded text.")
        self._text = text_

    @property
    def lbs(self):
        return self._lbs

    @lbs.setter
    def lbs(self, labels: List[List[str]]):
        assert len(self._text) == len(labels), ValueError("The number of text & labels instances does not match!")
        logger.warning("Setting label instances. Need to run `encode_text_and_lbs` to update encoded labels.")
        self._lbs = labels

    @property
    def n_insts(self):
        return len(self._encoded_texts.input_ids)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self._encoded_texts.items()}
        if self._lbs:
            item['labels'] = torch.tensor(self._lbs[idx])
        return item

    def encode_text(self,
                    config: BertCxConfig,
                    substitute_overlength_text: Optional[bool] = False):
        """
        Encode tokens so that they match the BERT data format

        Parameters
        ----------
        config: configuration file
        substitute_overlength_text: substitute overlength sentences by dummy tokens to maintain sentence numbers;
                                    should only be used during test without labels

        Returns
        -------
        self (BertClassificationDataset)
        """
        assert self._text, ValueError("Need to specify text")
        # logger.info("Encoding BERT text")

        tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)

        # exclude over-length instances
        encoded_token_lens = np.array([len(tokenizer.tokenize(txt, add_special_tokens=True))
                                       for txt in self._text])

        if not substitute_overlength_text:
            valid_instances = encoded_token_lens < config.max_length
            self._text = np.asarray(self._text, dtype=object)[valid_instances].tolist()
            if self._lbs:
                self._lbs = np.asarray(self._lbs, dtype=object)[valid_instances].tolist()
        else:
            assert not self._lbs, AssertionError('Labels should not be specified if '
                                                 '`substitute_overlength_text` is True!')

            invalid_instances = np.where(encoded_token_lens >= config.max_length)[0]
            for idx in invalid_instances:
                self._text[idx] = '<OVERLENGTH>'

        # logger.info('Encoding sentences into BERT tokens')
        self._encoded_texts = tokenizer(self._text,
                                        padding='max_length',
                                        max_length=config.max_length,
                                        truncation=True)

        return self

    def select(self, ids: Union[List[int], np.ndarray, torch.Tensor]):
        """
        Select a subset of dataset

        Parameters
        ----------
        ids: instance indices to select

        Returns
        -------
        A BertClassificationDataset consists of selected items
        """
        if np.max(ids) >= self.n_insts:
            logger.error("Invalid indices: exceeding the dataset size!")
            raise ValueError('Invalid indices: exceeding the dataset size!')
        text_ = np.asarray(self._text, dtype=object)[ids].tolist()
        lbs_ = np.asarray(self._lbs, dtype=object)[ids].tolist() if self._lbs else None
        encodings_ = {key: np.asarray(val, dtype=object)[ids].tolist() for key, val in self._encoded_texts.items()}
        # logger.warning("Need to run `encode_text` on the selected subset.")
        return BertClassificationDataset(text_, lbs_, BatchEncoding(encodings_))

    def load_file(self,
                  file_dir: str,
                  config: Optional[BertCxConfig] = None) -> "BertClassificationDataset":
        """
        Load data from disk

        Parameters
        ----------
        file_dir: the directory of the file. In JSON or PT
        config: chmm configuration; Optional to make function testing easier.

        Returns
        -------
        self (BertClassificationDataset)
        """

        file_dir = os.path.normpath(file_dir)
        logger.info(f'Loading data from {file_dir}')

        if file_dir.endswith('.json'):
            sentence_list, label_list = load_data_from_json(file_dir, config)
        else:
            logger.error(f"Unsupported data type: {file_dir}")
            raise TypeError(f"Unsupported data type: {file_dir}")

        self._text = sentence_list
        self._lbs = label_list
        logger.info(f'Data loaded from {file_dir}.')

        return self


def load_data_from_json(file_dir: str, config: Optional = None):
    """
    Load data stored in the current data format.


    Parameters
    ----------
    file_dir: file directory
    config: configuration

    """
    with open(file_dir, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    # Load meta if exist
    file_loc = os.path.split(file_dir)[0]
    meta_dir = os.path.join(file_loc, 'meta.json')

    if not os.path.isfile(meta_dir):
        logger.error('Meta file does not exist!')
        raise FileNotFoundError('Meta file does not exist!')

    with open(meta_dir, 'r', encoding='utf-8') as f:
        meta_dict = json.load(f)

    label_types = meta_dict['label_types']

    sentence_list = list()
    label_list = list()

    for i in range(len(data_dict)):
        data = data_dict[str(i)]
        sentence_list.append(data['text'])
        label_list.append(data['label'])

    # update config
    if config:
        config.label_types = label_types

    if config and config.debug:
        return sentence_list[:100], label_list[:100]
    return sentence_list, label_list
