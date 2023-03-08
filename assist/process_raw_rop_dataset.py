import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Union

from rdkit import Chem
from descriptastorus.descriptors import rdNormalizedDescriptors

from transformers import (
    HfArgumentParser,
    set_seed,
)

from seqlbtoolkit.io import set_logging, logging_args, save_json

Molecule = Union[str, Chem.Mol]
logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- IO arguments ---
    data_path: str = field(
        metadata={'help': 'Directory to datasets'}
    )
    output_dir: str = field(
        metadata={'help': "where to save model outputs."}
    )
    seed: Optional[int] = field(
        default=42, metadata={'help': 'random seed.'}
    )

    def __post_init__(self):
        pass


def main(args: Arguments):
    
    df = pd.read_csv(args.data_path)
    
    logger.info("Generating normalized rdkit fingerprints.")
    
    fingerprints = [np.array(rdkit_2d_features_normalized_generator(mol)) for mol in df['smiles_monomer']]
    # replace nan values
    fingerprints = [np.where(np.isnan(fingerprint), 0, fingerprint) for fingerprint in fingerprints]
    
    ids = np.arange(len(df))
    np.random.shuffle(ids)
    
    training_features = [fingerprints[idx] for idx in ids[:round(len(df) * 0.8)]]
    test_features = [fingerprints[idx] for idx in ids[round(len(df) * 0.8):]]

    training_smiles = [df["smiles_monomer"][idx] for idx in ids[:round(len(df) * 0.8)]]
    test_smiles = [df["smiles_monomer"][idx] for idx in ids[round(len(df) * 0.8):]]

    training_r_len = [df["1/length"][idx] for idx in ids[:round(len(df) * 0.8)]]
    test_r_len = [df["1/length"][idx] for idx in ids[round(len(df) * 0.8):]]

    training_roe = [df["ROE (kj/mol)"][idx] for idx in ids[:round(len(df) * 0.8)]]
    test_roe = [df["ROE (kj/mol)"][idx] for idx in ids[round(len(df) * 0.8):]]

    logger.info("Saving Data")
    training_data_dict = {
        "features": training_features,
        "smiles": training_smiles,
        "r_len": training_r_len,
        "targets": training_roe,
    }

    test_data_dict = {
        "features": test_features,
        "smiles": test_smiles,
        "r_len": test_r_len,
        "targets": test_roe,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(training_data_dict, os.path.join(args.output_dir, 'train.pt'))
    torch.save(test_data_dict, os.path.join(args.output_dir, 'test.pt'))

    logger.info("Saving Metadata")
    meta = {
        "d_feature": len(fingerprints[0]),
        "task": "regression",
        "is_rop_dataset": True
    }
    save_json(meta, os.path.join(args.output_dir, 'meta.json'))

    logger.info('Done.')


def rdkit_2d_features_normalized_generator(mol: Molecule) -> np.ndarray:
    """
    Generates RDKit 2D normalized features for a molecule.

    Parameters
    ----------
    mol: A molecule (i.e. either a SMILES string or an RDKit molecule).

    Returns
    -------
    An 1D numpy array containing the RDKit 2D normalized features.
    """
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    features = generator.process(smiles)[1:]
    return features


if __name__ == '__main__':

    _time = datetime.now().strftime("%m.%d.%y-%H.%M")
    _current_file_name = os.path.basename(__file__)
    if _current_file_name.endswith('.py'):
        _current_file_name = _current_file_name[:-3]

    # --- set up arguments ---
    parser = HfArgumentParser(Arguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        arguments, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        arguments, = parser.parse_args_into_dataclasses()

    if not getattr(arguments, "log_dir", None):
        arguments.log_dir = os.path.join('logs', f'{_current_file_name}', f'{_time}.log')

    set_logging(log_dir=arguments.log_dir)
    logging_args(arguments)

    set_seed(arguments.seed)

    try:
        main(args=arguments)
    except Exception as e:
        logger.exception(e)
        raise e
