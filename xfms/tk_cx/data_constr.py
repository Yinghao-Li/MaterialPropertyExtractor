import sys
sys.path.append('../..')

import json
import random
import numpy as np
from tqdm.auto import tqdm
from chemdataextractor.doc import Paragraph
from textspan import align_spans

from seqlbtoolkit.data import (
    txt_to_token_span,
    span_dict_to_list
)
from seqlbtoolkit.text import format_text


random.seed(42)


def get_data_from_multimodal(data_file='../data/polymer_names/multimodal_annos.jsonl'):
    with open(data_file, 'r', encoding='utf-8') as f:
        json_ls = [json.loads(jline) for jline in f.readlines()]

    tokens_list = list()
    spans_list = list()

    for json_l in tqdm(json_ls):
        ori_txt = json_l['text']
        txt = format_text(ori_txt)
        spans = json_l['spans']

        unlabeled_spans = [(s['start'], s['end']) for s in spans]
        aligned_spans = align_spans(unlabeled_spans, ori_txt, txt)
        lbs_dict = dict()
        for span, lb in zip(aligned_spans, spans):
            assert len(span) >= 1
            s = span[0][0]
            e = span[-1][1]
            lbs_dict[(s, e)] = lb['label']

        doc = Paragraph(txt)
        for sent in doc.sentences:
            tokens = [tk.text for tk in sent.tokens]
            txt_spans = dict()
            for (s, e), v in lbs_dict.items():
                if sent.start <= s and e <= sent.end:
                    txt_spans[(s - sent.start, e - sent.start)] = v
            tk_spans = txt_to_token_span(tokens, sent.text, txt_spans)

            tokens_list.append(tokens)
            spans_list.append(tk_spans)

    f_tokens_list = list()
    f_spans_list = list()

    # only keep the POLYMER entities
    for tks, spans in zip(tokens_list, spans_list):
        f_spans = dict()
        for (s, e), v in spans.items():
            if v == 'POLYMER':
                f_spans[(s, e)] = v
        if f_spans.values():
            f_tokens_list.append(tks)
            f_spans_list.append(f_spans)
        elif random.random() < 0.15:
            f_tokens_list.append(tks)
            f_spans_list.append(f_spans)

    return f_tokens_list, f_spans_list


def get_data_from_pet(data_file='../data/polymer_names/pet_annos.jsonl'):
    with open(data_file, 'r', encoding='utf-8') as f:
        json_ls = [json.loads(jline) for jline in f.readlines()]

    tokens_list = list()
    spans_list = list()

    for json_l in tqdm(json_ls):
        ori_txt = json_l['text']
        txt = format_text(ori_txt)
        lbs = json_l['labels']

        unlabeled_spans = [(s, e) for s, e, _ in lbs]
        aligned_spans = align_spans(unlabeled_spans, ori_txt, txt)
        lbs_dict = dict()
        for span, lb in zip(aligned_spans, lbs):
            assert len(span) >= 1
            s = span[0][0]
            e = span[-1][1]
            lbs_dict[(s, e)] = lb[2]

        doc = Paragraph(txt)

        for sent in doc.sentences:
            tokens = [tk.text for tk in sent.tokens]
            txt_spans = dict()
            for (s, e), v in lbs_dict.items():
                if sent.start <= s and e <= sent.end:
                    txt_spans[(s-sent.start, e-sent.start)] = v
            tk_spans = txt_to_token_span(tokens, sent.text, txt_spans)

            tokens_list.append(tokens)
            spans_list.append(tk_spans)

    f_tokens_list = list()
    f_spans_list = list()

    # only keep the POLYMER entities
    for tks, spans in zip(tokens_list, spans_list):
        if len(tks) > 200:
            continue

        f_spans = dict()
        for (s, e), v in spans.items():
            f_spans[(s, e)] = 'POLYMER'
        if f_spans.values():
            f_tokens_list.append(tks)
            f_spans_list.append(f_spans)
        elif random.random() < 0.15:
            f_tokens_list.append(tks)
            f_spans_list.append(f_spans)

    return f_tokens_list, f_spans_list


def main():

    tokens_list, spans_list = get_data_from_multimodal()

    pet_tokens, pet_spans = get_data_from_pet()
    inst_ids = list(range(len(pet_tokens)))
    random.shuffle(inst_ids)
    inst_ids = inst_ids[:2000]

    for inst_idx in inst_ids:
        tokens_list.append(pet_tokens[inst_idx])
        spans_list.append(pet_spans[inst_idx])

    inst_ids = list(range(len(tokens_list)))
    random.shuffle(inst_ids)

    train_ids = inst_ids[:int(np.ceil(0.7 * len(inst_ids)))]
    valid_ids = inst_ids[int(np.ceil(0.7 * len(inst_ids))): int(np.ceil(0.85 * len(inst_ids)))]
    test_ids = inst_ids[int(np.ceil(0.85 * len(inst_ids))):]

    train_list = list()
    for idx in train_ids:
        train_list.append({'text': tokens_list[idx], 'label': span_dict_to_list(spans_list[idx])})
    train_dict = {idx: inst for idx, inst in enumerate(train_list)}

    valid_list = list()
    for idx in valid_ids:
        valid_list.append({'text': tokens_list[idx], 'label': span_dict_to_list(spans_list[idx])})
    valid_dict = {idx: inst for idx, inst in enumerate(valid_list)}

    test_list = list()
    for idx in test_ids:
        test_list.append({'text': tokens_list[idx], 'label': span_dict_to_list(spans_list[idx])})
    test_dict = {idx: inst for idx, inst in enumerate(test_list)}

    with open('../data/polymer_names/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dict, f, ensure_ascii=False, indent=2)

    with open('../data/polymer_names/valid.json', 'w', encoding='utf-8') as f:
        json.dump(valid_dict, f, ensure_ascii=False, indent=2)

    with open('../data/polymer_names/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_dict, f, ensure_ascii=False, indent=2)

    sent_lens = [len(tks) for tks in tokens_list]
    max_len = np.max(sent_lens)
    mean_len = np.mean(sent_lens)

    meta_info = {
        'entity_types': ['POLYMER'],
        'train_size': len(train_ids),
        'valid_size': len(valid_ids),
        'test_size': len(test_ids),
        'max_length': int(max_len),
        'mean_length': float(mean_len),
        'num_labels': 3
    }
    with open('../data/polymer_names/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
