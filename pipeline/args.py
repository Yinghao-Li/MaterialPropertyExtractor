import logging
from typing import Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Arguments:

    # --- IO arguments ---
    doi_file_path: str = field(
        default='',
        metadata={"help": "The path to the files containing the DOIs of the articles to download."}
    )
    raw_article_dir: str = field(
        default='.',
        metadata={"help": "The path to the HTML/XML article file."}
    )
    processed_article_dir: str = field(
        default='.',
        metadata={"help": "Where to same the `.pt`-formatted processed articles"}
    )
    keyword_path: str = field(
        default='.',
        metadata={"nargs": '+', "help": "The path to keyword json files."}
    )
    extr_result_dir: Optional[str] = field(
        default='./output',
        metadata={"help": "Where to save the extracted property values"},
    )
    ner_dataset_dir: Optional[str] = field(
        default='./data/',
        metadata={'help': "Where to save the dataset used for training supervised NER models"}
    )
    material_name_model_dir: Optional[str] = field(
        default='./models/pet-mm-model', metadata={'help': 'directory of NER model for material name detection'}
    )
    property_model_dir: Optional[str] = field(
        default='', metadata={'help': 'directory of NER model for property & value detection'}
    )
    seqcx_model_dir: Optional[str] = field(
        default='', metadata={'help': 'sentence classification model directory'}
    )

    # --- task control arguments ---
    do_crawling: Optional[bool] = field(
        default=False, metadata={'help': "Whether crawl data from the web"}
    )
    do_parsing: Optional[bool] = field(
        default=False, metadata={'help': 'Whether parse articles'}
    )
    do_extraction: Optional[bool] = field(
        default=False, metadata={'help': 'Whether extract information'}
    )
    do_dataset_construction: Optional[bool] = field(
        default=False, metadata={'help': 'Whether construct dataset form the extraciton results'}
    )

    # --- information extraction arguments ---
    batch_size: Optional[int] = field(
        default=None, metadata={'help': 'model inference batch size. Leave None for original batch size'}
    )
    save_html: Optional[bool] = field(
        default=False, metadata={'help': 'Whether save HTML-formatted extraction results'}
    )
    save_jsonl: Optional[bool] = field(
        default=False, metadata={'help': 'Whether save JSONL-formatted extraction results'}
    )

    # --- dataset construction parameters ---
    valid_ratio: Optional[float] = field(
        default=0.15, metadata={'help': 'Ratio of validation instances'}
    )
    test_ratio: Optional[float] = field(
        default=0.15, metadata={'help': 'Ratio of test instances'}
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the directory of the log file. Set to '' to disable logging"}
    )
    seed: Optional[int] = field(
        default=42, metadata={'help': 'random seed'}
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )

    def __post_init__(self):
        if isinstance(self.keyword_path, str):
            self.keyword_path: List[str] = [self.keyword_path]
