# Convert positive and negative sentences to BERT data format
#
# last modified: 06/09/21

import os
import json
import random

from seqlbtoolkit.text import format_text


def convert_classification_data():

    sentence_dir = '../../classification-data'
    positive_files = 'positive.07-06-21.txt'
    negative_files = 'negative.txt'
    negative_distant_files = 'negative-samples.txt'
    fp_files = 'probably-false-positive.txt'

    save_dir = os.path.join('', '../data', f'train-fp')
    random.seed(0)

    if not os.path.isdir(save_dir):
        os.makedirs(os.path.abspath(save_dir))

    train_name = 'train.json'

    positive_dir = os.path.join(sentence_dir, positive_files)
    with open(positive_dir, 'r', encoding='utf-8') as f:
        positive_lines = f.readlines()

    negative_dir = os.path.join(sentence_dir, negative_files)
    with open(negative_dir, 'r', encoding='utf-8') as f:
        negative_lines = f.readlines()

    negative_distant_dir = os.path.join(sentence_dir, negative_distant_files)
    if os.path.isfile(negative_distant_dir):
        with open(negative_distant_dir, 'r', encoding='utf-8') as f:
            negative_distant_lines = f.readlines()

    fp_dir = os.path.join(sentence_dir, fp_files)
    if os.path.isfile(fp_dir):
        with open(fp_dir, 'r', encoding='utf-8') as f:
            fp_lines = f.readlines()
    random.shuffle(fp_lines)
    fp_lines = fp_lines[:800]

    positive_instances = list()
    for line in positive_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('$--$') and line.endswith('$--$'):
            continue
        positive_instances.append(format_text(line))

    negative_instances = list()
    for line in negative_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('$--$') and line.endswith('$--$'):
            continue
        negative_instances.append(format_text(line))

    negative_distant_instances = list()
    for line in negative_distant_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('$--$') and line.endswith('$--$'):
            continue
        negative_distant_instances.append(format_text(line))

    for line in fp_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('$--$') and line.endswith('$--$'):
            continue
        negative_distant_instances.append(format_text(line))

    x_insts = positive_instances + negative_instances + negative_distant_instances
    y_insts = [1] * len(positive_instances) + [0] * len(negative_instances) + [0] * len(negative_distant_instances)

    print(f'[INFO] Number of positive instances: {len(positive_instances)}\n'
          f'       Number of closely negative instances: {len(negative_instances)}\n'
          f'       Number of distantly negative instances: {len(negative_distant_instances)}')

    data_dict = dict()
    for i, (x, y) in enumerate(zip(x_insts, y_insts)):
        data_dict[i] = {
            'text': x,
            'label': y
        }

    with open(os.path.join(save_dir, train_name), 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    meta_info = {
        'label_types': [0, 1],
        'train_size': len(data_dict),
        'num_labels': 2
    }
    with open(os.path.join(save_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    convert_classification_data()
