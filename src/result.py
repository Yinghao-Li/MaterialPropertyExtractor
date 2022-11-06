from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Tuple

from seqlbtoolkit.base_model.eval import Metric
from seqlbtoolkit.text import remove_invalid_parenthesis
from .constants import *


class PropertyType(Enum):
    TEMPERATURE = 1
    ENTROPY = 2
    ENTHALPY = 3


@dataclass
class Property:
    material: Optional[str] = None
    property: Optional[str] = None
    value: Optional[Union[str, float, int]] = None
    unit: Optional[str] = None
    type: Optional[PropertyType] = None

    def from_span(self, text: str, mat_span: "PropertySpan") -> "Property":
        assert mat_span.value_span is not None, ValueError('Value not found!')
        # Do not need to convert values to float for now
        self.value = text[mat_span.value_span[0]: mat_span.value_span[1]].strip()

        if mat_span.unit_span is not None:
            self.unit = text[mat_span.unit_span[0]: mat_span.unit_span[1]].strip()

            # TODO: consider enhance this part with lsh fuzzy matching
            # if self.unit in TEMPERATURE_UNITS:
            #     self.type = PropertyType.TEMPERATURE
            # elif 'K' in self.unit:
            #     self.type = PropertyType.ENTROPY
            # else:
            #     self.type = PropertyType.ENTHALPY

        if mat_span.property_span is not None:
            ppt = None
            if isinstance(mat_span.property_span, list):
                prop_list = list()
                for cand_span in mat_span.property_span:
                    prop_list.append(text[cand_span[0]: cand_span[1]].strip())
                ppt = "; ".join(prop_list)
            elif isinstance(mat_span.property_span, tuple):
                ppt = text[mat_span.property_span[0]: mat_span.property_span[1]].strip()
            self.property = remove_invalid_parenthesis(ppt.strip(",.")) if ppt else None

        if mat_span.material_span is not None:
            m_name = None
            if isinstance(mat_span.material_span, list):
                mat_list = list()
                for cand_span in mat_span.material_span:
                    mat_list.append(text[cand_span[0]: cand_span[1]].strip())
                m_name = "; ".join(mat_list)
            elif isinstance(mat_span.material_span, tuple):
                m_name = text[mat_span.material_span[0]: mat_span.material_span[1]].strip()
            self.material = remove_invalid_parenthesis(m_name.strip(",.")) if m_name else None

        return self


class PropertySpan(Metric):
    def __init__(
            self,
            material_span: Optional[tuple] = None,
            property_span: Optional[tuple] = None,
            value_span: Optional[tuple] = None,
            unit_span: Optional[tuple] = None
    ):
        super().__init__()
        self.material_span = material_span
        self.property_span = property_span
        self.value_span = value_span
        self.unit_span = unit_span
        self.remove_attrs()

    def to_dict(self) -> Dict[Tuple[int, int], str]:
        """
        Convert class to dictionary {span(Tuple[int, int]): span-category}
        """
        span_dict = dict()
        for attr, lb in zip(['material_span', 'property_span', 'value_span', 'unit_span'],
                            [POLYMER_TAG, PROPERTY_TAG, NUMBER_TAG, UNIT_TAG]):
            span = getattr(self, attr, None)
            if span:
                span_dict[span] = lb
        return span_dict

    def __repr__(self):
        return f"material_span: {self.material_span}," \
               f" property_span: {self.property_span}," \
               f" value_span: {self.value_span}," \
               f" unit_span: {self.unit_span}"


def merge_list_of_property_spans(property_span_list: List[PropertySpan]) -> Dict[Tuple[int, int], str]:
    """
    Merge a list of property spans defined as PropertySpan

    Parameters
    ----------
    property_span_list: list of property spans

    Returns
    -------
    dict of span: tag
    """
    span_dict = dict()
    for property_span in property_span_list:
        span_dict.update(property_span.to_dict())
    return span_dict
