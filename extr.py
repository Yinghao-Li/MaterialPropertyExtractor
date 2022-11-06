import os
import sys
import logging
from datetime import datetime
from transformers import HfArgumentParser, set_seed
from seqlbtoolkit.io import set_logging, logging_args

from pipeline.args import Arguments
from pipeline.parser import parse
from pipeline.heuristic_extractor import extract
from pipeline.dataset_constructor import construct_dataset

logger = logging.getLogger(__name__)


def process(args: Arguments):
    if args.do_parsing:
        logger.info("Start parsing articles...")
        parse(args)
    if args.do_extraction:
        logger.info("Start extracting information from articles...")
        extract(args)
    if args.do_dataset_construction:
        logger.info("Constructing NER dataset from extraction results...")
        construct_dataset(args)
    logger.info("Done.")
    return None


if __name__ == '__main__':
    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        arguments, = parser.parse_args_into_dataclasses()

    # Setup logging
    if not getattr(arguments, "log_dir", None):
        arguments.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(arguments.log_dir)
    logger.setLevel(logging.INFO)
    logging_args(arguments)

    set_seed(arguments.seed)

    process(args=arguments)
