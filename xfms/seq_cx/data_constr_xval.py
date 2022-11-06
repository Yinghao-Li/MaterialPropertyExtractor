# Convert positive and negative sentences to BERT data format
#
# last modified: 06/09/21

import os
import json
import copy
import argparse
import random
from sklearn.model_selection import train_test_split
from seqlbtoolkit.text import format_text
from transformers import set_seed


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    return args


def convert_classification_data(args):

    random_seed = args.seed
    set_seed(random_seed)

    sentence_dir = '../../classification-data'
    positive_files = 'positive.07-06-21.txt'
    negative_files = 'negative.txt'
    negative_distant_files = 'negative-samples.txt'
    fp_files = 'probably-false-positive.txt'

    save_dir_distant = os.path.join('', '../data', f'distant-{random_seed}')

    if not os.path.isdir(save_dir_distant):
        os.makedirs(os.path.abspath(save_dir_distant))

    train_name = 'train.json'
    test_name = 'test.json'

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
    fp_lines = fp_lines[:500]

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

    x_insts = positive_instances + negative_instances
    y_insts = [1] * len(positive_instances) + [0] * len(negative_instances)
    neg_distant_x = negative_distant_instances
    neg_distant_y = [0] * len(negative_distant_instances)

    print(f'[INFO] Number of positive instances: {len(positive_instances)}\n'
          f'       Number of closely negative instances: {len(negative_instances)}\n'
          f'       Number of distantly negative instances: {len(negative_distant_instances)}')

    x_train, x_test, y_train, y_test = train_test_split(
        x_insts, y_insts, test_size=0.2, random_state=random_seed
    )

    x_neg_dist_train, x_neg_dist_test, y_neg_dist_train, y_neg_dist_test = train_test_split(
        neg_distant_x, neg_distant_y, test_size=0.2, random_state=random_seed
    )

    train_data = list()
    test_data = list()

    for x, y in zip(x_train, y_train):
        data_dict = {
            'sentence1': x,
            'label': y
        }
        train_data.append(json.dumps(data_dict, ensure_ascii=False))

    for x, y in zip(x_test, y_test):
        data_dict = {
            'sentence1': x,
            'label': y
        }
        test_data.append(json.dumps(data_dict, ensure_ascii=False))

    train_data_w_sample = copy.deepcopy(train_data)
    test_data_w_sample = copy.deepcopy(test_data)

    for x, y in zip(x_neg_dist_train, y_neg_dist_train):
        data_dict = {
            'sentence1': x,
            'label': y
        }
        train_data_w_sample.append(json.dumps(data_dict, ensure_ascii=False))

    for x, y in zip(x_neg_dist_test, y_neg_dist_test):
        data_dict = {
            'sentence1': x,
            'label': y
        }
        test_data_w_sample.append(json.dumps(data_dict, ensure_ascii=False))

    with open(os.path.join(save_dir_distant, train_name), 'w', encoding='utf-8') as f:
        for d in train_data_w_sample:
            f.write(d)
            f.write('\n')

    with open(os.path.join(save_dir_distant, test_name), 'w', encoding='utf-8') as f:
        for d in test_data_w_sample:
            f.write(d)
            f.write('\n')


if __name__ == '__main__':
    arguments = parse_args()
    convert_classification_data(arguments)
