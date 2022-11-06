import regex
import logging
from typing import Optional, List
from chempp.article import Article
from chempp.paragraph import DEFAULT_ANNO_SOURCE

from .constants import (
    Keywords,
    POLYMER_TAG,
    COMPOUND_TAG,
    VALID_SENT_TAG,
    PROPERTY_TAG,
    NUMBER_TAG,
    UNIT_TAG
)
from .result import merge_list_of_property_spans
from .rules.ner import (
    term_detector,
    unit_detector,
    number_detector,
    get_polymer_abbv,
    get_compound_name,
    validate_unit_spans,
    validate_term_spans
)
from .rules.re import (
    get_mat_property_spans,
    match_polymer_dict,
    link_mat_name
)

from xfms.utils import (
    load_ner_config_and_trainer,
    load_seqcx_config_and_trainer,
    get_seqcx_results,
    get_ner_results
)
from xfms.tk_cx.src.args import BertNERConfig
from xfms.tk_cx.src.train import BertNERTrainer
from xfms.seq_cx.src.args import BertCxConfig
from xfms.seq_cx.src.train import BertCxTrainer

logger = logging.getLogger(__name__)


class Annotator:
    def __init__(
            self,
            keyword_paths: Optional[List[str]] = None,
            material_name_ner_config: Optional[BertNERConfig] = None,
            material_name_ner_trainer: Optional[BertNERTrainer] = None,
            property_ner_config: Optional[BertNERConfig] = None,
            property_ner_trainer: Optional[BertNERTrainer] = None,
            seqcx_config: Optional[BertCxConfig] = None,
            seqcx_trainer: Optional[BertCxTrainer] = None
    ):
        if keyword_paths is not None:
            self.keyword_list = [Keywords(keyword_path) for keyword_path in keyword_paths]
        else:
            self.keyword_list = list()
        self._material_name_ner_config = material_name_ner_config
        self._material_name_ner_trainer = material_name_ner_trainer
        self._property_ner_config = property_ner_config
        self._property_ner_trainer = property_ner_trainer
        self._seqcx_config = seqcx_config
        self._seqcx_trainer = seqcx_trainer

    def load_bert_for_material_name(self, model_dir: str, batch_size: Optional[int] = None):
        """
        load pre-trained BERT for NER model

        Parameters
        ----------
        model_dir: model path
        batch_size: inference batch size

        Returns
        -------
        self
        """
        ner_config, ner_trainer = load_ner_config_and_trainer(model_dir, batch_size)
        self._material_name_ner_trainer = ner_trainer
        if self._material_name_ner_config is None:
            self._material_name_ner_config = ner_config
        return self

    def load_bert_for_property(self, model_dir: str, batch_size: Optional[int] = None):
        """
        load pre-trained BERT for property & value detection model

        Parameters
        ----------
        model_dir: model path
        batch_size: inference batch size

        Returns
        -------
        self
        """
        ner_config, ner_trainer = load_ner_config_and_trainer(model_dir, batch_size)
        self._property_ner_trainer = ner_trainer
        if self._property_ner_config is None:
            self._property_ner_config = ner_config
        return self

    def load_bert_for_seqcx(self, model_dir: str, batch_size: Optional[int] = None):
        """
        load pre-trained BERT for sequence classification

        Parameters
        ----------
        model_dir: model path
        batch_size: inference batch size

        Returns
        -------
        self
        """
        cx_config, cx_trainer = load_seqcx_config_and_trainer(model_dir, batch_size)
        self._seqcx_trainer = cx_trainer
        if self._seqcx_config is None:
            self._seqcx_config = cx_config
        return self

    def annotate_material_names_bert(self, article: Article, src_name: Optional[str] = DEFAULT_ANNO_SOURCE):
        """
        Annotate material names in the article with BERT-NER model

        Parameters
        ----------
        article: Article
            input article to annotate
        src_name: str
            the name of the BERT annotator.
            Assign value to this parameter if you want to distinguish it from other parameters.

        Returns
        -------
        article with BERT NER material name annotation
        """
        sent_list, tokens_list, inst_ids = article.get_sentences_and_tokens()

        ner_txt_span_list = get_ner_results(
            self._material_name_ner_trainer, self._material_name_ner_config, sent_list, tokens_list
        )
        for sent_id, ner_txt_spans in enumerate(ner_txt_span_list):
            if not ner_txt_spans:
                continue
            for ner_txt_span in ner_txt_spans:
                # The deep model sometimes catches pure numbers. Need to exclude these FPs
                try:
                    float(article[inst_ids[sent_id]].text[ner_txt_span[0]: ner_txt_span[1]])
                except ValueError:
                    article[inst_ids[sent_id]].anno[src_name][ner_txt_span] = POLYMER_TAG
            article[inst_ids[sent_id][0]].update_paragraph_anno()
        return article

    @staticmethod
    def annotate_material_names_heuristic(article: Article, src_name: Optional[str] = DEFAULT_ANNO_SOURCE):
        """
        Annotate material names in the article with rules rules

        Parameters
        ----------
        article: input article to annotate
        src_name: the name of the rules rules

        Returns
        -------
        article with rules rule material name annotations
        """
        sent_list, tokens_list, inst_ids = article.get_sentences_and_tokens()

        polymer_abbv = get_polymer_abbv(sent_list)
        polymer_abbv_re = [regex.escape(tk) for tk in polymer_abbv]
        compound_names = get_compound_name(sent_list)
        compound_names_re = [regex.escape(tk) for tk in compound_names]

        for para in article.paragraphs:
            sec_txt = para.text
            dict_spans = match_polymer_dict(sec_txt)
            for span in dict_spans:
                para.anno[src_name][span] = POLYMER_TAG

            if polymer_abbv_re:
                abbv_spans = term_detector(sec_txt, polymer_abbv_re)
                for span in abbv_spans:
                    para.anno[src_name][span] = POLYMER_TAG

            if compound_names_re:
                compound_spans = term_detector(sec_txt, compound_names_re)
                for span in compound_spans:
                    para.anno[src_name][span] = COMPOUND_TAG

            para.remove_anno_overlaps()
            para.update_sentence_anno()

        return article

    def link_property_annotations(self,
                                  article: Article,
                                  valid_sent_ids: List[int],
                                  src_name: Optional[str] = DEFAULT_ANNO_SOURCE):
        """
        Link property names, values, and units.

        Parameters
        ----------
        article: Article
            input article to annotate
        valid_sent_ids: int
            the indices of sentences that may contain desired information
        src_name: str
            The source name of valid sentence tagger

        Returns
        -------
        (Article, bool that indicate whether the article contain any positive sentences)
        """

        sent_list, _, inst_ids = article.get_sentences_and_tokens()
        mat_property_span_list = list()
        update_sent_ids = list()
        for sent_id in valid_sent_ids:
            mat_property_spans = get_mat_property_spans(sent_list[sent_id], self.keyword_list)
            valid_property_spans = list(filter(
                lambda mat: mat.property_span is not None and mat.value_span is not None and mat.unit_span is not None,
                mat_property_spans
            ))
            if valid_property_spans:
                update_sent_ids.append(sent_id)
                mat_property_span_list.append(valid_property_spans)

        valid_inst_ids = [inst_ids[idx] for idx in update_sent_ids]

        if not valid_sent_ids:
            return article, False

        for (sec_id, sent_idx), mat_property_spans in zip(valid_inst_ids, mat_property_span_list):
            art_sent = article[sec_id].sentences[sent_idx]
            article[sec_id].anno[src_name][(art_sent.start_idx, art_sent.end_idx)] = VALID_SENT_TAG
            article[sec_id, sent_idx].grouped_anno += mat_property_spans
            article[sec_id, sent_idx].anno = merge_list_of_property_spans(mat_property_spans)
            article[sec_id].update_paragraph_anno_group(sent_idx)
            article[sec_id].update_paragraph_anno(sent_idx)

        return article, True

    def find_valid_sentence_bert(self, article: Article):
        """
        Use BERT sentence classification model to detect sentences that contains property-value pairs

        Parameters
        ----------
        article: the input article to annotate

        Returns
        -------
        annotated article
        """
        sent_list, _, inst_ids = article.get_sentences_and_tokens()
        # get polymer names and predicted sentences
        valid_sent_ids = get_seqcx_results(self._seqcx_trainer, self._seqcx_config, sent_list)
        # make sure every sentence contain at least one number
        vs_ids = list()
        for idx in valid_sent_ids:
            if number_detector(sent_list[idx]):
                vs_ids.append(idx)
        valid_sent_ids = vs_ids

        article, flag = self.link_property_annotations(article, valid_sent_ids)
        return article, flag

    def annotate_property_bert(self, article: Article, src_name: Optional[str] = DEFAULT_ANNO_SOURCE):
        """
        Use BERT NER model to detect property-value-unit tuples

        Parameters
        ----------
        article: Article
            Input article to annotate
        src_name: str
            The source name of valid sentence tagger

        Returns
        -------
        annotated article
        """
        sent_list, tokens_list, inst_ids = article.get_sentences_and_tokens()
        # get polymer names and predicted sentences
        ner_txt_span_list = get_ner_results(
            self._property_ner_trainer,
            self._property_ner_config,
            sent_list,
            tokens_list
        )

        valid_sent_ids = list()
        for sent_id, ner_txt_spans in enumerate(ner_txt_span_list):

            if not ner_txt_spans:
                continue

            # put all spans into the article annotation results
            for ner_txt_span, tag in ner_txt_spans.items():
                article[inst_ids[sent_id]].anno[src_name][ner_txt_span] = tag
            article[inst_ids[sent_id][0]].update_paragraph_anno(sent_id)

            # skip sentences without both numbers and units
            span_tags = list(ner_txt_spans.values())
            if NUMBER_TAG in span_tags and UNIT_TAG in span_tags:
                valid_sent_ids.append(sent_id)

        article, flag = self.link_property_annotations(article, valid_sent_ids)
        return article, flag

    def annotate_property_heuristic(self, article: Article):
        """
        Use rules rules to detect property-value pairs

        Parameters
        ----------
        article: the input article to annotate

        Returns
        -------
        annotated article
        """
        sent_list, _, inst_ids = article.get_sentences_and_tokens()

        valid_sent_ids = list()
        for idx, sent in enumerate(sent_list):
            for keyword in self.keyword_list:
                unit_spans = unit_detector(sent, keyword.units_pattern)
                term_spans = term_detector(sent, keyword.terms_pattern)
                number_spans = number_detector(sent)

                valid_spans = validate_unit_spans(unit_spans=unit_spans, number_spans=number_spans)
                valid_spans += validate_term_spans(
                    text=sent, term_spans=term_spans, number_spans=number_spans, unit_spans=unit_spans
                )
                if valid_spans:
                    valid_sent_ids.append(idx)

        article, flag = self.link_property_annotations(article, valid_sent_ids)
        return article, flag

    @staticmethod
    def link_material_name_to_property(article: Article):
        """
        link material name to property with rules rules

        Parameters
        ----------
        article: Article
            input article annotated with material name and properties

        Returns
        -------
        article with name-property links
        """
        return link_mat_name(article)
