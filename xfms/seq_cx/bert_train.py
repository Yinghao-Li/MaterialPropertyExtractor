# coding=utf-8
""" Fully supervised BERT-NER """

import logging
import os
import sys
import gc
import torch
from datetime import datetime

from transformers import (
    HfArgumentParser,
    set_seed,
)

from seqlbtoolkit.io import set_logging, logging_args

from .src.data import BertClassificationDataset
from .src.train import BertCxTrainer
from .src.args import BertCxArguments, BertCxConfig

logger = logging.getLogger(__name__)


def bert_train(args: BertCxArguments):
    # setup logging
    set_logging(log_dir=args.log_dir)

    logging_args(args)
    set_seed(args.seed)
    config = BertCxConfig().from_args(args)

    training_dataset = valid_dataset = test_dataset = None
    if args.train_file:
        logger.info('Loading training dataset...')
        training_dataset = BertClassificationDataset().load_file(
            file_dir=args.train_file,
            config=config
        ).encode_text(config=config)
        logger.info(f'Training dataset loaded, length={len(training_dataset)}')

    if args.valid_file:
        logger.info('Loading validation dataset...')
        valid_dataset = BertClassificationDataset().load_file(
            file_dir=args.valid_file,
            config=config
        ).encode_text(config=config)
        logger.info(f'Validation dataset loaded, length={len(valid_dataset)}')

    if args.test_file:
        logger.info('Loading test dataset...')
        test_dataset = BertClassificationDataset().load_file(
            file_dir=args.test_file,
            config=config
        ).encode_text(config=config)
        logger.info(f'Test dataset loaded, length={len(test_dataset)}')

    # create output dir if it does not exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(os.path.abspath(args.output_dir))

    bert_trainer = BertCxTrainer(
        config=config,
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    ).initialize_trainer()

    if args.train_file:
        logger.info("Start training Bert.")
        training_results = bert_trainer.train()
    else:
        bert_trainer.load(args.output_dir, load_optimizer_and_scheduler=True)
        training_results = None

    if args.test_file:
        logger.info("Start testing Bert.")

        test_results = bert_trainer.test()

        logger.info("Test results:")
        for k, v in test_results.items():
            logger.info(f"\t{k}: {v:.4f}")
    else:
        test_results = None

    result_file = os.path.join(args.output_dir, 'bert-results.txt')
    logger.info(f"Writing results to {result_file}")
    with open(result_file, 'w') as f:
        if training_results is not None:
            for i, results in enumerate(training_results):
                f.write(f"[Epoch {i + 1}]\n")
                for k, v in results.items():
                    f.write(f"\t{k}: {v:.4f}\n")
        if test_results is not None:
            f.write(f"[Test]\n")
            for k, v in test_results.items():
                f.write(f"\t{k}: {v:.4f}\n")

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(BertCxArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        bert_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        bert_args, = parser.parse_args_into_dataclasses()

    if bert_args.log_dir is None:
        bert_args.log_dir = os.path.join('logs', f'{_current_file_name}.{_time}.log')

    bert_train(args=bert_args)
