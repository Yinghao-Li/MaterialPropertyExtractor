import logging
import numpy as np

from typing import Optional
from seqlbtoolkit.data import label_to_span, token_to_txt_span

from .tk_cx.src.args import BertNERConfig
from .tk_cx.src.train import BertNERTrainer
from .tk_cx.src.data import BertNERDataset

from .seq_cx.src.args import BertCxConfig
from .seq_cx.src.train import BertCxTrainer
from .seq_cx.src.data import BertClassificationDataset


logger = logging.getLogger(__name__)


__all__ = [
    "load_ner_config_and_trainer",
    "load_seqcx_config_and_trainer",
    "get_seqcx_results",
    "get_ner_results"
]


def load_ner_config_and_trainer(model_dir: str, batch_size: Optional[int] = None):
    ner_config = BertNERConfig().load(model_dir)
    if batch_size:
        ner_config.batch_size = batch_size
    ner_trainer = BertNERTrainer(ner_config).load(model_dir)
    return ner_config, ner_trainer


def load_seqcx_config_and_trainer(model_dir: str, batch_size: Optional[int] = None):
    seqcx_config = BertCxConfig().load(model_dir)
    if batch_size:
        seqcx_config.batch_size = batch_size
    seqcx_trainer = BertCxTrainer(seqcx_config).load(model_dir)
    return seqcx_config, seqcx_trainer


def get_seqcx_results(classifier_trainer, classifier_config, sent_list):
    # load classification dataset
    classification_dataset = BertClassificationDataset(
        text=sent_list
    ).encode_text(classifier_config, True)

    # classify sentences
    classification_output = classifier_trainer.predict(classification_dataset)
    classification_lbs, _ = classification_output

    valid_sent_ids = np.where(np.asarray(classification_lbs) == 1)[0].tolist()

    return valid_sent_ids


def get_ner_results(ner_model_trainer, ner_model_config, sent_list, tokens_list):
    # load ner dataset
    ner_dataset = BertNERDataset(
        text=tokens_list
    ).encode_text(ner_model_config, True)

    # predict named entities
    ner_output = ner_model_trainer.predict(ner_dataset)
    ner_lbs, ner_probs = ner_output

    # format outputs
    ner_spans = [label_to_span(lbs) for lbs in ner_lbs]
    ner_txt_spans = [token_to_txt_span(tks, txt, spans) for tks, txt, spans in
                     zip(tokens_list, sent_list, ner_spans)]

    return ner_txt_spans
