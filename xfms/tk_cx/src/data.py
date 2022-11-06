import os
import json
import logging
import numpy as np

from typing import List, Optional, Union
from seqlbtoolkit.data import (
    entity_to_bio_labels,
    span_list_to_dict,
    span_to_label
)

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BatchEncoding
)

from .args import BertNERConfig

logger = logging.getLogger(__name__)


class BertNERDataset(torch.utils.data.Dataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 lbs: Optional[List[List[str]]] = None,
                 encoded_texts: Optional[BatchEncoding] = BatchEncoding(),
                 encoded_lbs: Optional[List[List[int]]] = None,
                 token_masks: Optional[List[List[int]]] = None,
                 ):
        super().__init__()
        self._text = text
        self._lbs = lbs
        # splitted text so that every sentence is within maximum length when they are converted to BERT tokens
        self._encoded_texts = encoded_texts
        self._encoded_lbs = encoded_lbs if encoded_lbs is not None else list()
        # mask out sub-tokens and paddings
        self._token_masks = token_masks if token_masks is not None else list()

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text_: List[List[str]]):
        logger.warning("Setting text instances. Need to run `encode_text` or `encode_text_and_lbs` "
                       "to update encoded text.")
        self._text = text_

    @property
    def lbs(self):
        return self._lbs

    @lbs.setter
    def lbs(self, labels: List[List[str]]):
        assert len(self._text) == len(labels), ValueError("The number of text & labels instances does not match!")
        for txt, lbs_ in zip(self._text, labels):
            assert len(txt) == len(lbs_), ValueError("The lengths of text & labels instances does not match!")
        logger.warning("Setting label instances. Need to run `encode_text_and_lbs` to update encoded labels.")
        self._lbs = labels

    @property
    def token_masks(self):
        return np.asarray(self._token_masks)

    @property
    def encoded_lbs(self):
        return np.asarray(self._encoded_lbs)

    @property
    def n_insts(self):
        return len(self._encoded_texts.input_ids)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self._encoded_texts.items() if key != 'offset_mapping'}
        item['token_masks'] = torch.tensor(self._token_masks[idx])
        if self._encoded_lbs:
            item['labels'] = torch.tensor(self._encoded_lbs[idx])
        return item

    def encode_text(self,
                    config: BertNERConfig,
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
        self (BertNERDataset)
        """
        assert self._text, ValueError("Need to specify text")
        # logger.info("Encoding BERT text")

        tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)

        # exclude over-length instances
        encoded_token_lens = np.array([len(tokenizer.tokenize(' '.join(txt), add_special_tokens=True))
                                       for txt in self._text])

        if not substitute_overlength_text:
            valid_instances = encoded_token_lens < config.max_length
            self._text = np.asarray(self._text, dtype=object)[valid_instances].tolist()
        else:
            invalid_instances = np.where(encoded_token_lens >= config.max_length)[0]
            for idx in invalid_instances:
                self._text[idx] = ['<OVERLENGTH>']

        # logger.info('Encoding sentences into BERT tokens')
        self._encoded_texts = tokenizer(self._text,
                                        is_split_into_words=True,
                                        return_offsets_mapping=True,
                                        padding='max_length',
                                        max_length=config.max_length,
                                        truncation=True)

        token_masks = list()
        for doc_offset in self._encoded_texts.offset_mapping:
            arr_offset = np.array(doc_offset)

            # create an empty array of False
            masks = np.zeros(len(doc_offset), dtype=np.bool)
            masks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = True
            token_masks.append(masks)
        self._token_masks = token_masks

        return self

    def encode_text_and_lbs(self, config: BertNERConfig):
        """
        Encode tokens and labels so that they match the BERT data format

        Parameters
        ----------
        config: configuration file

        Returns
        -------
        self (BertNERDataset)
        """
        assert self._text and self._lbs, ValueError("Need to specify text and labels")

        # logger.info("Encoding BERT text and labels")

        tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name_or_path)

        # exclude over-length instances
        encoded_token_lens = np.array([len(tokenizer.tokenize(' '.join(txt), add_special_tokens=True))
                                       for txt in self._text])
        valid_instances = encoded_token_lens < config.max_length
        self._text = np.asarray(self._text, dtype=object)[valid_instances].tolist()
        self._lbs = np.asarray(self._lbs, dtype=object)[valid_instances].tolist()

        # logger.info('Encoding sentences into BERT tokens')
        self._encoded_texts = tokenizer(self._text,
                                        is_split_into_words=True,
                                        return_offsets_mapping=True,
                                        padding='max_length',
                                        max_length=config.max_length,
                                        truncation=True)

        labels = [[config.lb2idx[lb] for lb in lbs] for lbs in self._lbs]

        encoded_labels = list()
        token_masks = list()

        # logger.info('Aligning labels to encoded text')
        for idx, (doc_labels, doc_offset) in enumerate(zip(labels, self._encoded_texts.offset_mapping)):
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

            # create an empty array of False
            masks = np.zeros(len(doc_offset), dtype=np.bool)
            masks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = True
            token_masks.append(masks)
        self._encoded_lbs = encoded_labels
        self._token_masks = token_masks

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
        logger.warning("Need to run `encode_text` or `encode_text_and_lbs` on the selected subset.")
        return BertNERDataset(text_, lbs_)

    def load_file(self,
                  file_dir: str,
                  config: Optional[BertNERConfig] = None) -> "BertNERDataset":
        """
        Load data from disk

        Parameters
        ----------
        file_dir: the directory of the file. In JSON or PT
        config: chmm configuration; Optional to make function testing easier.

        Returns
        -------
        self (BERTNERDataset)
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

    bio_labels = entity_to_bio_labels(meta_dict['entity_types'])

    sentence_list = list()
    span_list = list()

    for i in range(len(data_dict)):
        data = data_dict[str(i)]
        sentence_list.append(data['text'])
        span_list.append(data['label'])

    span_dicts = list()
    for spans in span_list:
        span_dict = span_list_to_dict(spans)
        span_dicts.append(span_dict)

    lbs_list = [span_to_label(sps, tks) for sps, tks in zip(span_dicts, sentence_list)]

    # update config
    if config:
        config.entity_types = meta_dict['entity_types']
        config.bio_label_types = bio_labels

    if config and config.debug:
        return sentence_list[:100], lbs_list[:100]
    return sentence_list, lbs_list
