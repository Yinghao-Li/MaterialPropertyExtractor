import numpy as np
from typing import List, Optional

from chempp.article import Article

from ..result import PropertyType, Property
from .ner import term_detector, unit_detector, number_detector

__all__ = [
    "heuristic_criteria",
    "group_rop_properties",
    "check_keyword",
    "get_property_info"
]

REL_SCORE_MAPPING = {
    'c1': 4,
    'c2': 2,
    'c3': 0
}


def heuristic_criteria(sents):
    text = ' '.join(sents).lower()
    tc_exist = True if term_detector(text, ['ceiling temperature']) else False
    entropy_exist = True if term_detector(text, ['entropy']) else False
    enthalpy_exist = True if term_detector(text, ['enthalpy']) else False

    if tc_exist and entropy_exist and enthalpy_exist:
        return 'c1'
    elif tc_exist and (entropy_exist or enthalpy_exist):
        return 'c2'
    elif tc_exist or entropy_exist or enthalpy_exist:
        return 'c3'
    else:
        return None


def check_keyword(article: Article, term_patterns: str):
    sent_list, _, _ = article.get_sentences_and_tokens()
    if not term_detector(' '.join(sent_list), term_patterns):
        return False

    criterion = heuristic_criteria(sent_list)
    if not criterion:
        return False
    return criterion


def get_entity_values(sent: str, entropy_patterns, enthalpy_patterns, tc_patterns):

    entropy_unit_spans = unit_detector(sent, entropy_patterns)
    enthalpy_unit_spans = unit_detector(sent, enthalpy_patterns)
    entropy_span_start_pos = [s[0] for s in entropy_unit_spans]
    enthalpy_unit_spans = [s for s in enthalpy_unit_spans if s[0] not in entropy_span_start_pos]

    temperature_unit_spans = unit_detector(sent, tc_patterns)
    number_spans = number_detector(sent)

    tc_values = list()
    for tu_span in temperature_unit_spans:
        us, ue = tu_span
        for number_span in number_spans:
            ns, ne = number_span
            if np.abs(us - ne) < 3:
                tc_values.append(sent[ns: ue])

    enthalpy_values = list()
    entropy_values = list()
    for eu_span in entropy_unit_spans:
        es, ee = eu_span
        for number_span in number_spans:
            ns, ne = number_span
            if np.abs(es - ne) < 3:
                entropy_values.append(sent[ns: ee])
    for eu_span in enthalpy_unit_spans:
        es, ee = eu_span
        for number_span in number_spans:
            ns, ne = number_span
            if np.abs(es - ne) < 3:
                enthalpy_values.append(sent[ns: ee])

    return tc_values, entropy_values, enthalpy_values


def append_entropy_value(info_list, entropy_values, sent):
    for v in entropy_values:
        info_list.append({
            'Tc': '',
            'ΔH': '',
            'ΔS': v,
            'sentence': sent,
        })


def append_enthalpy_value(info_list, enthalpy_values, sent):
    for v in enthalpy_values:
        info_list.append({
            'Tc': '',
            'ΔH': v,
            'ΔS': '',
            'sentence': sent,
        })


def append_tc_value(info_list, ct_values, sent):
    for v in ct_values:
        info_list.append({
            'Tc': v,
            'ΔH': '',
            'ΔS': '',
            'sentence': sent,
        })


def group_rop_properties(sent: str):
    tc_values, entropy_values, enthalpy_values = get_entity_values(sent)
    info_list = list()

    if len(tc_values) == 1 and len(enthalpy_values) == 1 and len(entropy_values) == 1:
        info_list.append({
            'Tc': tc_values[0],
            'ΔH': enthalpy_values[0],
            'ΔS': entropy_values[0],
            'sentence': sent,
        })
    elif len(tc_values) == 1 and len(enthalpy_values) == 1:
        info_list.append({
            'Tc': tc_values[0],
            'ΔH': enthalpy_values[0],
            'ΔS': '',
            'sentence': sent,
        })
        append_entropy_value(info_list, entropy_values, sent)
    elif len(tc_values) == 1 and len(entropy_values) == 1:
        info_list.append({
            'Tc': tc_values[0],
            'ΔH': '',
            'ΔS': entropy_values[0],
            'sentence': sent,
        })
        append_enthalpy_value(info_list, entropy_values, sent)
    elif len(enthalpy_values) == 1 and len(entropy_values) == 1:
        info_list.append({
            'Tc': '',
            'ΔH': enthalpy_values[0],
            'ΔS': entropy_values[0],
            'sentence': sent,
        })
        append_tc_value(info_list, tc_values, sent)
    else:
        append_tc_value(info_list, tc_values, sent)
        append_enthalpy_value(info_list, enthalpy_values, sent)
        append_entropy_value(info_list, entropy_values, sent)

    return info_list


def get_property_info(article: Article, criterion: Optional[str] = None):
    sentence_id = 0
    prev_sent = ''

    info_list = list()
    for para in article.paragraphs:
        if not para.grouped_anno:
            continue
        for pg_anno in para.grouped_anno:
            properties = Property().from_span(para.text, pg_anno)
            if properties.type == PropertyType.TEMPERATURE:
                p_type = 'Tc'
            elif properties.type == PropertyType.ENTROPY:
                p_type = 'ΔS'
            elif properties.type == PropertyType.ENTHALPY:
                p_type = 'ΔH'
            else:
                raise ValueError(f'Unknown property type: {properties.type}')

            sent = para.get_sentence_by_char_idx(pg_anno.value_span[0]).text
            if sent != prev_sent:
                sentence_id += 1
                prev_sent = sent
            info_list.append({
                'sentence': sent,
                'sentence-id': sentence_id,
                'material': properties.material if properties.material is not None else '',
                'property': properties.property,
                'value': ' '.join([str(properties.value), properties.unit]),
                'type': p_type
            })
    score = get_article_score(info_list, criterion)

    for info_dict in info_list:
        info_dict['reliability'] = float(score)
    return info_list


def get_article_score(property_value_list: List[dict], criterion: Optional[str] = None):

    if criterion is not None:
        reliability = REL_SCORE_MAPPING[criterion]
    else:
        reliability = 0

    for property_values in property_value_list:
        p_type = property_values['type']

        tc_values = int(p_type == "Tc")
        entropy_values = int(p_type == 'ΔS')
        enthalpy_values = int(p_type == 'ΔH')

        reliability += 0.8 * (tc_values and entropy_values and enthalpy_values)
        reliability += 0.3 * (tc_values + entropy_values + enthalpy_values == 2)
        reliability += 0.1 * (tc_values + entropy_values + enthalpy_values)
    return reliability
