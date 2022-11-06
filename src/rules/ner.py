import regex
import textspan
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from chemdataextractor.doc import Paragraph
from typing import List, Union

from seqlbtoolkit.data import (
    respan,
    txt_to_token_span,
    token_to_txt_span,
    sort_tuples_by_element_idx,
    merge_overlapped_spans
)


def unit_detector(text, units: Union[str, list, tuple]):
    """
    Detect unit keywords or patterns

    Parameters
    ----------
    text: input text
    units: units to detect. Could be a list of keywords or a regex pattern string

    Returns
    -------
    detected text-level spans
    """
    if isinstance(units, (list, tuple)):
        units_pattern = '|'.join(units)
    elif isinstance(units, (str, regex.Pattern)):
        units_pattern = units
    else:
        raise TypeError(f'Undefined type {type(units)} for units')
    spans = regex.finditer(r'([ \d\p{Ps}\p{Pe}\'"\u2018-\u201d])(%s)'
                           r'($|[ \p{Ps}\p{Pe}\'"\u2018-\u201d,.])' % units_pattern, text)
    detected_spans = [(m.span(2)[0], m.span(2)[1]) for m in spans]
    detected_spans = list(set(detected_spans))
    detected_spans = sort_tuples_by_element_idx(detected_spans)

    return detected_spans


def term_detector(text, terms):
    """
    Detect terms keywords or patterns

    Parameters
    ----------
    text: input text
    terms: terms to detect. Could be a list of keywords or a regex pattern string

    Returns
    -------
    detected text-level spans
    """
    if isinstance(terms, (list, tuple)):
        terms_pattern = '|'.join(terms)
    elif isinstance(terms, (str, regex.Pattern)):
        terms_pattern = terms
    else:
        raise TypeError(f'Undefined type {type(terms)} for terms')
    spans = regex.finditer(r'(^|[ \p{Ps}\'"\u2018-\u201d])(%s)'
                           r'($|[ \p{Ps}\p{Pe}\'"\u2018-\u201d,.])' % terms_pattern, text)
    detected_spans = [(m.span(2)[0], m.span(2)[1]) for m in spans]
    detected_spans = list(set(detected_spans))
    detected_spans = sort_tuples_by_element_idx(detected_spans)

    return detected_spans


def number_detector(text):
    cwt = ChemWordTokenizer()
    tokens = cwt.tokenize(text)
    cde_text = ' '.join(tokens)

    spans = regex.finditer(r"(?:^|[\p{Ps} =<>]) *(-* *\d+(?:.\d+)*((?:-|±|,| |to|or|and)+\d+(?:.\d+)*)*)"
                           r"(?:[.]*$|[ \p{Ps}\p{Pe}])", cde_text)
    spans = [(m.span(1)[0], m.span(1)[1]) for m in spans]
    spans = list(set(spans))

    detected_spans = list()
    token_spans = txt_to_token_span(tokens, cde_text, spans)
    for span, token_span in zip(spans, token_spans):
        ts = token_span[0]
        prev_tokens = ' '.join(tokens[ts - 3 if ts - 3 > 0 else 0: ts]).lower().strip().replace('.', '')
        element_keywords = 'figure|fig|table|tb|eq|equ|equation|section|sec|alg|algorithm|§|chapter|ch'
        if ts == 0 or not regex.findall(
                r'(^|[\p{Ps} ])(%s)([.]*$|[ \p{Ps}\p{Pe}])' % element_keywords, prev_tokens
        ):
            detected_spans.append(span)

    detected_spans = [s[0] for s in textspan.align_spans(detected_spans, cde_text, text)]
    detected_spans = sort_tuples_by_element_idx(detected_spans)
    return detected_spans


def get_nn_chucks(pos_tags: List[str], tokens: List[str], sent: str):
    nn_chunks = list()
    chunk_start = chunk_end = None
    for i, (tag, tk) in enumerate(zip(pos_tags, tokens)):
        if (tag in ["NN", 'NNP', 'NNS']) or tk in ["°", "(", ")", "[", "]"]:
            if chunk_start is None:
                chunk_start = i
            else:
                chunk_end = i + 1
        else:
            if chunk_start is not None and chunk_end is not None:
                if tokens[chunk_start] in ["(", "["]:
                    chunk_start += 1
                if tokens[chunk_end] in [")", "]"]:
                    chunk_end -= 1
                if chunk_end > chunk_start:
                    nn_chunks.append((chunk_start, chunk_end))
                chunk_start = chunk_end = None
            elif chunk_start is not None and chunk_end is None:
                if tokens[chunk_start] not in ["(", "["]:
                    nn_chunks.append((chunk_start, chunk_start + 1))
                chunk_start = chunk_end = None

    if chunk_start is not None and chunk_end is not None:
        nn_chunks.append((chunk_start, chunk_end))

    word_tokens = sent.split(' ')
    word_nn_spans = respan(tokens, word_tokens, nn_chunks)
    word_nn_spans = merge_overlapped_spans(word_nn_spans)

    text_spans = token_to_txt_span(word_tokens, sent, word_nn_spans)
    text_spans = sort_tuples_by_element_idx(text_spans)

    return text_spans


def get_polymer_abbv(sents: Union[str, List[str]]) -> List[str]:
    if isinstance(sents, list):
        sents = ' '.join(sents)

    para = Paragraph(sents)
    potential_poly_abbv_list = list()
    for abbvs, full_names, abbv_type in para.abbreviation_definitions:
        if not abbv_type == 'CM':
            continue

        abbv = abbvs[0]
        if full_names[0].startswith('poly'):
            if abbv not in potential_poly_abbv_list:
                potential_poly_abbv_list.append(abbv)
            continue

        if f'P {abbv}' not in potential_poly_abbv_list:
            potential_poly_abbv_list.append(f'P {abbv}')
        if f'P{abbv}' not in potential_poly_abbv_list:
            potential_poly_abbv_list.append(f'P{abbv}')

    return potential_poly_abbv_list


def get_compound_name(sents: Union[str, List[str]]) -> List[str]:
    if isinstance(sents, list):
        sents = ' '.join(sents)

    para = Paragraph(sents)
    comp_list = list()
    for abbvs, full_names, abbv_type in para.abbreviation_definitions:
        if not abbv_type == 'CM':
            continue

        abbv = abbvs[0]
        full_name = ' '.join(full_names)
        if full_name.startswith('poly'):
            continue
        if full_name not in comp_list:
            comp_list.append(full_name)
        if abbv not in comp_list:
            comp_list.append(abbv)

    return comp_list


def validate_unit_spans(unit_spans, number_spans):
    # check if unit_spans are valid
    valid_spans = list()
    for unit_span in unit_spans:
        s = unit_span[0]
        for number_span in number_spans:
            ne = number_span[1]
            if abs(s - ne) <= 5:
                valid_spans.append(unit_span)
                break
    return valid_spans


def validate_term_spans(text, term_spans, number_spans, unit_spans):
    if (not term_spans) or (not number_spans) or (not unit_spans):
        return []

    para = Paragraph(text)
    sent_ranges = [(sent.start, sent.end) for sent in para]

    # check if term_spans are valid
    valid_spans = list()
    for term_span in term_spans:
        number_valid_flag = False
        unit_valid_flag = False
        s = term_span[0]
        e = term_span[1]
        sent_range = (0, 1)
        # check which sentence the target span belongs to.
        for sent_range in sent_ranges:
            if sent_range[0] <= s < sent_range[1]:
                break
        # make sure Tc refers to the ceiling temperature
        if text[s-1 if s > 0 else 0: e+1 if e+1 < len(text) else len(text)] == '(Tc)':
            if 'ceiling' not in text[sent_range[0]: sent_range[1]]:
                continue
        for number_span in number_spans:
            ns = number_span[0]
            if sent_range[0] <= ns < sent_range[1]:
                number_valid_flag = True
                break
        for unit_span in unit_spans:
            us = unit_span[0]
            if sent_range[0] <= us < sent_range[1] and number_spans[0][1] < us:
                unit_valid_flag = True
                break
        if number_valid_flag and unit_valid_flag:
            valid_spans.append(term_span)
    return valid_spans
