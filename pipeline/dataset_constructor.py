import os
import glob
import json
import random
import logging
import numpy as np

from tqdm.auto import tqdm
from natsort import natsorted
from chemdataextractor.doc import Paragraph

from seqlbtoolkit.text import format_text
from seqlbtoolkit.data import txt_to_token_span
from seqlbtoolkit.io import save_json

from src.constants import (
    PROPERTY_TAG,
    NUMBER_TAG,
    UNIT_TAG
)

from .args import Arguments

logger = logging.getLogger(__name__)


def construct_dataset(args: Arguments):

    file_dir = os.path.join(args.extr_result_dir, 'jsonl-files')
    assert os.path.exists(file_dir), FileNotFoundError(f"{file_dir} not found!")

    path_list = natsorted(glob.glob(os.path.join(file_dir, '*.jsonl')))

    data_list = list()
    debug_idx = 0
    for file_path in tqdm(path_list):

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        inst_list = list()
        for line in lines:
            inst_list.append(json.loads(line))

        for anno_inst in inst_list:

            # --- convert raw text to paragraphs and sentences ---
            article = anno_inst['text']
            paragraphs = [a for a in article.split('\n')]

            para_list = list()
            positions = list()  # list of tuple (start, end)
            curr_pos = 0
            for paragraph in paragraphs:
                if paragraph:
                    para_list.append(format_text(paragraph))
                    positions.append((curr_pos, curr_pos + len(paragraph) + 1))
                curr_pos += len(paragraph) + 1

            sentences = list()
            sent_positions = list()
            for para, (s, e) in zip(para_list, positions):
                cde_para = Paragraph(para)
                for sent in cde_para.sentences:
                    sentences.append(sent)
                    sent_positions.append((s + sent.start, s + sent.end))

            # --- Process heuristically annotated labels ---
            lbs = anno_inst['label']

            # keep spans with specific text
            spans_list = list()
            for tag in [PROPERTY_TAG, NUMBER_TAG, UNIT_TAG]:
                spans_list.append([(k[0], k[1]) for k in lbs if k[2] == tag])
            property_spans, number_spans, unit_spans = spans_list

            # convert span tuples to span annotation dicts {(start, end): entity type}
            lb_dicts = dict()
            p_dicts = {p_span: PROPERTY_TAG for p_span in property_spans}
            n_dicts = {n_span: NUMBER_TAG for n_span in number_spans}
            u_dicts = {u_span: UNIT_TAG for u_span in unit_spans}

            # merge annotation dicts with differenct values (entity types)
            for d in [p_dicts, n_dicts, u_dicts]:
                lb_dicts.update(d)

            # convert article-level text spans to sentence-level text spans
            sorted_lb_dicts = {s[0]: s[1] for s in sorted(lb_dicts.items(), key=lambda k: k[0][0])}
            sent_start_idx = 0
            tk_sent_ids = list()
            lb_spans = list()
            for (tk_s, tk_e), tag in sorted_lb_dicts.items():
                for sent_i, (sent_s, sent_e) in enumerate(sent_positions[sent_start_idx:], sent_start_idx):
                    if tk_s > sent_s and tk_e <= sent_e:
                        sent_start_idx = sent_i
                        tk_sent_ids.append(sent_i)
                        lb_spans.append((tk_s - sent_s, tk_e - sent_s, tag))
                        break

            # convert text spans to token spans
            sent_id_2_inst = dict()
            for sent_id, lb_span in zip(tk_sent_ids, lb_spans):

                sent = sentences[sent_id]
                if sent_id not in sent_id_2_inst:
                    tks = [tk.text for tk in sent.tokens]
                    sent_id_2_inst[sent_id] = {'text': tks, 'label': list()}
                else:
                    tks = sent_id_2_inst[sent_id]['text']

                tk_span_dict = txt_to_token_span(tks, sent.text, {(lb_span[0], lb_span[1]): lb_span[2]})
                tk_span_list = list(map(lambda x: (x[0][0], x[0][1], x[1]), tk_span_dict.items()))
                sent_id_2_inst[sent_id]['label'] += tk_span_list

            for v in sent_id_2_inst.values():
                data_list.append(v)

            retry_count = 0
            filtered_neg_sent_ids = list()
            while True:
                n_sent_to_sample = len(sent_id_2_inst) - len(filtered_neg_sent_ids)

                if not n_sent_to_sample:
                    break

                neg_sents_ids = random.choices(range(len(sentences)), k=n_sent_to_sample)
                for idx in neg_sents_ids:
                    if len(sentences[idx].tokens) <= 5 or idx in filtered_neg_sent_ids:
                        continue
                    filtered_neg_sent_ids.append(idx)

                if retry_count == 2:
                    break

                retry_count += 1

            neg_sents = [sentences[idx] for idx in filtered_neg_sent_ids]
            for neg_sent in neg_sents:
                tks = [tk.text for tk in neg_sent.tokens]
                data_list.append({'text': tks, 'label': list()})

        if args.debug:
            debug_idx += 1
        if debug_idx == 5:
            break

    # partition and save datasets
    random.shuffle(data_list)
    n_valid = round(len(data_list) * args.valid_ratio)
    n_test = round(len(data_list) * args.test_ratio)
    valid_list = data_list[: n_valid]
    test_list = data_list[n_valid: n_valid + n_test]
    train_list = data_list[n_valid + n_test:]

    for file_name, partition in zip(['train', 'valid', 'test'], [train_list, valid_list, test_list]):
        partition_dict = {str(idx): inst for idx, inst in enumerate(partition)}
        save_json(obj=partition_dict,
                  path=os.path.join(args.ner_dataset_dir, f'{file_name}.json'),
                  collapse_level=3)

    # calculate and save metadata
    meta = {
        'entity_types': [PROPERTY_TAG, NUMBER_TAG, UNIT_TAG],
        'train_size': len(train_list),
        'valid_size': len(valid_list),
        'test_size': len(test_list),
        'max_length': np.max([len(inst['text']) for inst in data_list]).item(),
        'mean_length': np.mean([len(inst['text']) for inst in data_list]).item(),
        'num_labels': 7
    }
    save_json(obj=meta, path=os.path.join(args.ner_dataset_dir, 'meta.json'))

    return None
