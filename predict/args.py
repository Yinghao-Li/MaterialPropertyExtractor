import os
import torch
import logging
from typing import Optional
from dataclasses import dataclass, field

from transformers.file_utils import cached_property, torch_required
from seqlbtoolkit.training.config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class Arguments:
    """
    Arguments regarding the training of Neural hidden Markov Model
    """

    # --- wandb parameters ---
    wandb_api_key: Optional[str] = field(
        default=None, metadata={'help': 'The API key that indicates your wandb account.'
                                        'Can be found here: https://wandb.ai/settings'}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={'help': 'name of the wandb project.'}
    )
    wandb_name: Optional[str] = field(
        default=None, metadata={'help': 'wandb model name.'}
    )

    # --- IO arguments ---
    data_dir: Optional[str] = field(
        default='', metadata={'help': 'Directory to datasets'}
    )
    output_dir: Optional[str] = field(
        default='./output', metadata={'help': "where to save model outputs."}
    )
    plot_dir: Optional[str] = field(
        default=None, metadata={'help': 'where to save plots'}
    )
    model_dir: Optional[str] = field(
        default=None, metadata={'help': "The folder where the trained model is saved"}
    )
    model_name: Optional[str] = field(
        default='model', metadata={'help': "Name of the model"}
    )
    overwrite_output: Optional[bool] = field(
        default=False, metadata={'help': 'Whether overwrite existing outputs.'}
    )

    # --- Model Arguments ---
    dropout: Optional[float] = field(
        default=0.1, metadata={'help': "Dropout ratio."}
    )

    # --- Training Arguments ---
    batch_size: Optional[int] = field(
        default=32, metadata={'help': "Batch size."}
    )
    n_epochs: Optional[int] = field(
        default=50, metadata={'help': "How many epochs to train the model."}
    )
    lr: Optional[float] = field(
        default=1e-4, metadata={'help': "Learning Rate."}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    debug: Optional[bool] = field(
        default=False, metadata={"help": "Debugging mode with fewer training data"}
    )
    save_preds: Optional[bool] = field(
        default=False, metadata={"help": "Whether save test predicts into disk."}
    )

    # --- Model Arguments ---
    n_hidden_layers: Optional[int] = field(
        default=8, metadata={'help': 'How many hidden layers are included in the model'}
    )
    d_hidden: Optional[int] = field(
        default=512, metadata={'help': 'Hidden dimensionality'}
    )

    # --- Device Arguments ---
    no_cuda: Optional[bool] = field(
        default=False, metadata={"help": "Disable CUDA even when it is available."}
    )
    num_workers: Optional[int] = field(
        default=0, metadata={"help": 'The number of threads to process dataset.'}
    )

    def __post_init__(self):
        self.apply_wandb = self.wandb_project and self.wandb_name and not self.debug

        if not self.plot_dir:
            self.plot_dir = os.path.join(self.output_dir, 'plots')

    # The following three functions are copied from transformers.training_args
    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        if self.no_cuda or not torch.cuda.is_available():
            device = torch.device("cpu")
            self._n_gpu = 0
        else:
            device = torch.device("cuda")
            self._n_gpu = 1

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    @torch_required
    def n_gpu(self) -> int:
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        _ = self._setup_devices
        return self._n_gpu


class Config(Arguments, BaseConfig):

    d_feature = None
    label_types = None
    task = "classification"
    is_rop_dataset = False

    @property
    def n_lbs(self):
        return len(self.label_types) if self.label_types is not None else 1
