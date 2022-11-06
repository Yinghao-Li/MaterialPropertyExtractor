import os
import glob
import torch
import logging
import pandas as pd

from natsort import natsorted
from seqlbtoolkit.text import substring_mapping
from seqlbtoolkit.io import init_dir

from chempp.constants import CHAR_TO_HTML_LBS

from src.constants import (
    POLYMER_TAG,
    VALID_SENT_TAG
)
from src.pipeline import Annotator
from src.result import Property

from .args import Arguments

logger = logging.getLogger(__name__)


def extract(args: Arguments):

    annotator = Annotator(keyword_paths=args.keyword_path).load_bert_for_material_name(model_dir=args.material_name_model_dir)

    if args.processed_article_dir.endswith('.pt'):
        path_list = [args.processed_article_dir]
    else:
        path_list = natsorted(glob.glob(os.path.join(args.processed_article_dir, '*.pt')))

    output_html_dir = output_jsonl_dir = None
    if args.save_html:
        output_html_dir = os.path.normpath(os.path.join(args.extr_result_dir, 'html-files'))
        init_dir(output_html_dir)
    if args.save_jsonl:
        output_jsonl_dir = os.path.normpath(os.path.join(args.extr_result_dir, 'jsonl-files'))
        init_dir(output_jsonl_dir)

    global_property_dict = dict()
    for file_path in path_list:

        logger.info(f'Processing file {os.path.basename(file_path)}')

        try:
            article = torch.load(file_path)

            article, has_property = annotator.annotate_property_heuristic(article)
            if not has_property:
                continue
            article = annotator.annotate_material_names_bert(article)
            article = annotator.annotate_material_names_heuristic(article)
            article = annotator.link_material_name_to_property(article)

            property_list = list()
            for para in article.paragraphs:
                for result_group in para.grouped_anno:
                    property_list.append(Property().from_span(para.text, result_group))

            if not property_list:
                continue

            global_property_dict[article.doi] = property_list

            if args.save_html:
                logger.info(f"Saving HTML results...")
                output_file_name = f"{substring_mapping(article.doi, CHAR_TO_HTML_LBS)}.html"
                output_file = os.path.normpath(os.path.join(output_html_dir, output_file_name))
                article.save_html(output_file,
                                  tags_to_highlight=[POLYMER_TAG, VALID_SENT_TAG],
                                  tags_to_present=[VALID_SENT_TAG])

            if args.save_jsonl:
                logger.info(f"Saving JSONL results...")
                output_file_name = f"{substring_mapping(article.doi, CHAR_TO_HTML_LBS)}.jsonl"
                output_file = os.path.normpath(os.path.join(output_jsonl_dir, output_file_name))
                article.save_jsonl(output_file)

        except Exception as e:
            logger.exception(e)

    if global_property_dict:

        logger.info("Grouping extraction results")
        kws = ['material', 'property', 'value', 'unit']
        result_dict = {kw: list() for kw in ['doi'] + kws}
        for doi, property_list in global_property_dict.items():
            for ppt in property_list:
                result_dict['doi'].append(doi)
                for kw in kws:
                    result_dict[kw].append(getattr(ppt, kw))
        pd.DataFrame(result_dict).to_csv(os.path.join(args.extr_result_dir, 'extr_results.csv'))

    return None
