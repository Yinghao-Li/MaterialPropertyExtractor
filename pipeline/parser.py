import os
import glob
import torch
import logging

from natsort import natsorted
from seqlbtoolkit.text import substring_mapping

from chempp.article_constr import (
    parse_html,
    parse_xml
)
from chempp.constants import CHAR_TO_HTML_LBS

from .args import Arguments

logger = logging.getLogger(__name__)


def parse(args: Arguments):

    path_list = natsorted(glob.glob(os.path.join(args.raw_article_dir, '*')))
    os.makedirs(args.processed_article_dir, exist_ok=True)

    for file_path in path_list:

        logger.info(f'Parsing file {os.path.basename(file_path)}')
        try:
            if file_path.lower().endswith('html'):
                article, component_check = parse_html(file_path)
            elif file_path.lower().endswith('xml'):
                article, component_check = parse_xml(file_path)
            else:
                raise ValueError('Unsupported file type!')

            # Check components
            if not component_check.abstract:
                logger.warning(f'{file_path} does not have abstract!')
            if not component_check.sections:
                logger.warning(f'{file_path} does not have HTML sections!')
            if not (component_check.abstract or component_check.sections):
                logger.warning("Parsed article is emtpy!")
                continue

            logger.info("Saving results...")
            # Save parsed files
            output_name = f"{substring_mapping(article.doi, CHAR_TO_HTML_LBS)}.pt"
            save_path = os.path.normpath(os.path.join(args.processed_article_dir, output_name))

            torch.save(article, save_path)

        except Exception as e:
            logger.exception(e)

    return None
