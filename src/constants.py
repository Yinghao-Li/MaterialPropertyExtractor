import regex
import json

__all__ = [
    "Keywords",
    "COMP_CHARS", 'EQUAL_CHARS', 'POLYMER_NAMES_PATTERN', "POLYMER_NAMES_RE",
    "POLYMER_TAG", "VALID_SENT_TAG", "COMPOUND_TAG", "PROPERTY_TAG", "NUMBER_TAG", "UNIT_TAG"
]


class Keywords:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            keyword_dict = json.load(f)

        self.terms = [regex.escape(kw) for kw in keyword_dict['terms']]
        self.terms_pattern = keyword_dict.get('terms-pattern', []) if keyword_dict.get('terms-pattern', []) else \
            '|'.join(self.terms)
        self.units = [regex.escape(kw) for kw in keyword_dict['units']]
        self.units_pattern = keyword_dict.get('units-pattern', []) if keyword_dict.get('units-pattern', []) else \
            '|'.join(self.units)


# Rampi had also told us he was interested in collecting data about the 'degradation temperature'
# this is not directly related to the ceiling temperature but Rampi believes they are correlated, so I included it
# Terms relating to degradation temperature:
# 'Td','degradation temperature', 'T d'

# these terms are included:'enthalpy of formation', 'entropy of formation', 'free energy of formation',
# because the ceiling temperature can be computed from them even if the paper is not directly about ceiling temp

# these terms: 'chemical recycling to monomer', 'CRM','chemical recycling','reversible ring opening','
# Dainton', 'Dainton Equation','Ivin','Dainton-Ivin'
# because papers including them are likely to mention ceiling temperature

import os
import pathlib
current_dir = pathlib.Path(__file__).parent.parent.resolve()
list_dir = './dependency/polymer-names.json'
with open(os.path.join(current_dir, list_dir), 'r', encoding='utf-8') as f:
    polymer_name_list = json.load(f)
polymer_name_list = [x for x in polymer_name_list if len(x) < 50]

POLYMER_NAMES_RE = [regex.escape(name) for name in polymer_name_list]
POLYMER_NAMES_PATTERN = regex.compile('|'.join(POLYMER_NAMES_RE))


COMP_CHARS = ["~", "≈", "=", "≅", "≤", "≥", "⩽", "⩾", "<", ">"]
EQUAL_CHARS = ["~", "≈", "=", "≅"]

POLYMER_TAG = 'POLYMER'
COMPOUND_TAG = 'COMPOUND'
PROPERTY_TAG = 'PROPERTY'
UNIT_TAG = 'UNIT'
NUMBER_TAG = 'NUM'
VALID_SENT_TAG = 'VALID-SENT'
