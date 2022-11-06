import regex
import textspan
from typing import List, Tuple
from chemdataextractor.doc import Sentence

from chempp.article import Article
from seqlbtoolkit.data import (
    respan_text,
    merge_overlapped_spans,
    sort_tuples_by_element_idx
)

from .ner import (
    get_nn_chucks,
    term_detector,
    number_detector,
    unit_detector
)

from ..result import PropertySpan
from ..constants import *


def associate_number_unit(unit_spans: List[tuple], number_spans: List[tuple]) -> List[PropertySpan]:
    property_list = list()

    for n_span in number_spans:
        n_end = n_span[1]
        for unit_span in unit_spans:
            u_start = unit_span[0]
            if abs(n_end - u_start) <= 5:
                property_list.append(PropertySpan(
                    value_span=n_span,
                    unit_span=unit_span
                ))
                break
    return property_list


def match_polymer_dict(text: str) -> List[Tuple[int, int]]:
    spans = POLYMER_NAMES_PATTERN.finditer(text)
    detected_spans = [m.span() for m in spans]
    return detected_spans


# noinspection PyTypeChecker
def append_property_names_to_mat_spans(
        mat_spans: List[PropertySpan],
        sent: str,
        property_term_pattern: str,
        use_equality_mark_approx=False
        ):
    """
    TODO: should revise this function. Some method seems unreasonable now.
    """
    s = Sentence(sent)
    tokens = [tk.text for tk in s.tokens]
    nn_spans = get_nn_chucks(s.tags, tokens, sent)
    nn_spans += term_detector(sent, property_term_pattern)
    nn_spans = sort_tuples_by_element_idx(nn_spans)
    nn_spans = merge_overlapped_spans(nn_spans)

    joint_tks = ' '.join(tokens)
    operator_spans = term_detector(joint_tks, EQUAL_CHARS)
    operator_spans = respan_text(joint_tks, sent, operator_spans)

    for mat_span in mat_spans:
        assert mat_span.value_span is not None, ValueError('Span of property values is not found!')

        # associate property names with values by the equality mark "="
        ns = mat_span.value_span[0]

        if use_equality_mark_approx:
            for operator_span in operator_spans:
                os, oe = operator_span

                if abs(oe - ns) <= 3:
                    for nn_span in nn_spans:
                        nne = nn_span[1]
                        if abs(os - nne) <= 3:
                            mat_span.property_span = nn_span
                            break
                    break

        # matching sub-words
        if mat_span.property_span is None:
            for nn_span in nn_spans:
                if nn_span[0] > ns:
                    break
                nn_text = sent[nn_span[0]:nn_span[1]]
                if regex.findall(property_term_pattern, nn_text):
                    mat_span.property_span = nn_span

    return mat_spans


def get_mat_property_spans(sent: str, keywords: List[Keywords]) -> List[PropertySpan]:
    """
    Get polymerization properties and values in a sentence
    """
    number_spans = number_detector(sent)

    property_span_list = list()
    for keyword in keywords:
        unit_spans = unit_detector(sent, keyword.units_pattern)
        unit_spans = sort_tuples_by_element_idx(unit_spans)
        unit_spans = textspan.remove_span_overlaps(unit_spans)

        property_spans = associate_number_unit(unit_spans, number_spans)
        property_spans = append_property_names_to_mat_spans(
            property_spans,
            sent,
            property_term_pattern=keyword.terms_pattern
        )
        property_span_list += property_spans

    return property_span_list


def link_mat_name(article: Article) -> Article:
    for para in article.paragraphs:
        if not para.grouped_anno:
            continue
        for anno_group in para.grouped_anno:
            if anno_group.material_span:
                continue
            value_span = anno_group.value_span
            sent = para.get_sentence_by_char_idx(value_span[0])

            cand_span = dict()
            for (s, e), value in sent.all_anno.items():
                if value == POLYMER_TAG and sent.text[s: e] not in cand_span.values():
                    cand_span[(s, e)] = sent.text[s: e]

            if cand_span:  # find material spans in the same sentence
                if len(cand_span) == 0:
                    (s, e) = list(cand_span.keys())[0]
                    anno_group.material_span = (s + sent.start_idx, e + sent.start_idx)
                else:
                    anno_group.material_span = list()
                    for (s, e) in cand_span.keys():
                        anno_group.material_span.append((s + sent.start_idx, e + sent.start_idx))
            else:  # no material spans in the property sentence
                sorted_spans = sort_tuples_by_element_idx(list(para.all_anno.keys()))
                cached_span = None
                for span in sorted_spans:
                    if para.all_anno[span] != POLYMER_TAG:
                        continue
                    if span[0] > value_span[0]:
                        break
                    cached_span = span
                anno_group.material_span = cached_span

            # If we already find some polymer mentions, exit
            if anno_group.material_span is not None:
                continue

            cand_span = dict()
            for (s, e), value in sent.all_anno.items():
                if value == COMPOUND_TAG and sent.text[s: e] not in cand_span.values():
                    cand_span[(s, e)] = sent.text[s: e]

            if cand_span:  # find material spans in the same sentence
                if len(cand_span) == 0:
                    (s, e) = list(cand_span.keys())[0]
                    anno_group.material_span = (s + sent.start_idx, e + sent.start_idx)
                else:
                    anno_group.material_span = list()
                    for (s, e) in cand_span.keys():
                        anno_group.material_span.append((s + sent.start_idx, e + sent.start_idx))
            else:  # no material spans in the property sentence
                sorted_spans = sort_tuples_by_element_idx(list(para.all_anno.keys()))
                cached_span = None
                for span in sorted_spans:
                    if para.all_anno[span] != COMPOUND_TAG:
                        continue
                    if span[0] > value_span[0]:
                        break
                    cached_span = span
                anno_group.material_span = cached_span

    return article
