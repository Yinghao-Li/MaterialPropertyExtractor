import os
import logging
import numpy as np

from tqdm.auto import tqdm
from typing import Optional
from collections import OrderedDict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_scheduler,
    default_data_collator
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.modeling_outputs import SequenceClassifierOutput

from .args import BertCxConfig
from .data import BertClassificationDataset

logger = logging.getLogger(__name__)


class BertCxTrainer:
    """
    Bert trainer used for training BERT for token classification (sequence labeling)
    """
    def __init__(self,
                 config: BertCxConfig,
                 collate_fn: Optional[PreTrainedModel] = default_data_collator,
                 model: Optional[PreTrainedModel] = None,
                 tokenzier: Optional[PreTrainedTokenizer] = None,
                 training_dataset: Optional[BertClassificationDataset] = None,
                 valid_dataset: Optional[BertClassificationDataset] = None,
                 test_dataset: Optional[BertClassificationDataset] = None,
                 optimizer=None,
                 lr_scheduler=None):

        self._model = model
        self._tokenizer = tokenzier
        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        logger.warning("Updating BertTrainer.config")
        self._config = x

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def training_dataset(self):
        return self._training_dataset

    @training_dataset.setter
    def training_dataset(self, dataset):
        logger.warning("BertNERDataset.training_dataset is updated!")
        self._training_dataset = dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @valid_dataset.setter
    def valid_dataset(self, dataset):
        logger.warning("BertNERDataset.valid_dataset is updated!")
        self._valid_dataset = dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset):
        logger.warning("BertNERDataset.test_dataset is updated!")
        self._test_dataset = dataset

    def initialize_trainer(self, model=None, tokenizer=None, optimizer=None, lr_scheduler=None):
        """
        Initialize necessary components for training

        Returns
        -------
        the initialized trainer
        """
        self.set_model(model, tokenizer)
        self.set_optimizer_scheduler(optimizer, lr_scheduler)
        return self

    def set_datasets(self, training=None, valid=None, test=None):
        """
        Set bert trainer datasets

        Parameters
        ----------
        training: training dataset
        valid: validation dataset
        test: test dataset

        Returns
        -------
        self
        """
        if training:
            self.training_dataset = training
        if valid:
            self.valid_dataset = valid
        if test:
            self.test_dataset = test
        return self

    def set_model(self, model=None, tokenizer=None):
        """
        Initialize BERT model by given model/default

        Parameters
        ----------
        model: input BERT model
        tokenizer: input tokenizer

        Returns
        -------
        self
        """
        if model is not None:
            if self._model is not None:
                logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                               f"It will be overwritten by input!")
            self._model = model
            assert tokenizer is not None, ValueError("The tokenizer has to be assigned along with the model.")
            self._tokenizer = tokenizer
        else:
            if self._model is not None:
                logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                               f"It will be re-initialized by default!")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self._config.bert_model_name_or_path,
                num_labels=self._config.n_lbs
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._config.bert_model_name_or_path)
        return self

    def set_optimizer_scheduler(self, optimizer=None, lr_scheduler=None):
        """
        create optimizer and scheduler

        Parameters
        ----------
        optimizer: input optimizer
        lr_scheduler: input learning rate scheduler

        Returns
        -------
        self (BertTrainer)
        """
        if optimizer is not None:
            if self._optimizer is not None:
                logger.warning(f"The original optimizer {type(self._optimizer)} in {type(self)} is not None. "
                               f"It will be overwritten by input!")
            self._optimizer = optimizer
        else:
            if self._optimizer is not None:
                logger.warning(f"The original optimizer {type(self._optimizer)} in {type(self)} is not None."
                               f"It will be re-initialized by default!")
            # The following codes are modified from transformers.Trainer.create_optimizer_and_scheduler
            decay_parameters = get_parameter_names(self._model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self._model.named_parameters() if n in decay_parameters],
                    "weight_decay": self._config.weight_decay,
                },
                {
                    "params": [p for n, p in self._model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self._optimizer = AdamW(optimizer_grouped_parameters, lr=self._config.learning_rate)

        if lr_scheduler is not None:
            if self._lr_scheduler is not None:
                logger.warning(f"The original learning rate scheduler {type(self._lr_scheduler)} in {type(self)} "
                               f"is not None. It will be overwritten by input!")
            self._lr_scheduler = lr_scheduler
        else:
            if self._lr_scheduler is not None:
                logger.warning(f"The original learning rate scheduler {type(self._lr_scheduler)} in {type(self)} "
                               f"is not None. It will be re-initialized by default!")
            # The following codes are modified from transformers.Trainer.create_optimizer_and_scheduler
            assert self._training_dataset, AttributeError("Need to define training set to initialize lr scheduler.")
            if not self._config.batch_gradient_descent:
                num_update_steps_per_epoch = int(np.ceil(
                    len(self._training_dataset) * (1-self._config.validation_ratio) / self._config.batch_size
                ))
            else:
                num_update_steps_per_epoch = 1

            num_warmup_steps = int(np.ceil(
                num_update_steps_per_epoch * self._config.warmup_ratio * self._config.num_train_epochs))
            num_training_steps = int(np.ceil(num_update_steps_per_epoch * self._config.num_train_epochs))
            self._lr_scheduler = get_scheduler(
                self._config.lr_scheduler_type,
                self._optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        return self

    def train(self):
        self._model.to(self._config.device)

        n_insts = len(self._training_dataset)
        n_fold = int(np.floor(1 / self._config.validation_ratio))
        val_size = int(np.floor(n_insts * self._config.validation_ratio))
        inst_ids = np.arange(n_insts)

        valid_result_list = list()
        best_f1 = 0
        val_f1_cache = 0
        tolerance_epoch = 0

        # ----- start training process -----
        logger.info("Start training BERT...")
        for epoch_i in range(self._config.num_train_epochs):

            logger.info("------")
            logger.info(f"Epoch {epoch_i + 1} of {self._config.num_train_epochs}")

            if self.valid_dataset is not None:
                training_dataloader = self.get_dataloader(self._training_dataset, shuffle=True)
                valid_dataset = self.valid_dataset
                epoch_mod = 0
            else:
                # Prepare cross validation dataset
                epoch_mod = epoch_i % n_fold

                if epoch_mod == 0:
                    inst_ids = np.arange(n_insts)
                    np.random.shuffle(inst_ids)
                val_ids = inst_ids[val_size * epoch_mod: val_size * (epoch_mod + 1)]
                train_ids = np.setdiff1d(inst_ids, val_ids)

                training_dataset = self._training_dataset.select(train_ids)
                valid_dataset = self._training_dataset.select(val_ids)
                training_dataloader = self.get_dataloader(training_dataset, shuffle=True)

            train_loss = self.training_step(training_dataloader, self._optimizer, self._lr_scheduler)
            logger.info("Training loss: %.4f" % train_loss)

            valid_results = self.evaluate(valid_dataset)

            logger.info("Validation results:")
            for k, v in valid_results.items():
                logger.info(f"\t{k}: {v:.4f}")

            # ----- save model -----
            if self.valid_dataset is not None:
                if valid_results['f1'] >= best_f1:
                    self.save()
                    logger.info("Checkpoint Saved!\n")
                    best_f1 = valid_results['f1']
                    tolerance_epoch = 0
                else:
                    tolerance_epoch += 1
            else:
                if epoch_mod == (n_fold - 1):
                    val_f1_cache /= n_fold
                    if val_f1_cache >= best_f1:
                        logger.info(f'Saving checkpoint...')
                        self.save()
                        best_f1 = val_f1_cache
                        tolerance_epoch = 0
                    else:
                        tolerance_epoch += 1
                    val_f1_cache = 0
                else:
                    val_f1_cache += valid_results['f1']

            # ----- log history -----
            valid_result_list.append(valid_results)
            if tolerance_epoch > self._config.num_valid_tolerance:
                logger.info("Training stopped because of exceeding tolerance")
                break

        # retrieve the best state dict
        self.load()
        return valid_result_list

    def training_step(self, data_loader, optimizer, lr_scheduler):
        train_loss = 0
        num_samples = 0

        self._model.train()
        optimizer.zero_grad()

        for inputs in tqdm(data_loader):
            # get data
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._config.device)
            batch_size = len(inputs['input_ids'])
            num_samples += batch_size

            # training step
            outputs: SequenceClassifierOutput = self._model(**inputs)
            loss = self.compute_loss(outputs, inputs)
            loss.backward()
            # track loss
            train_loss += loss.detach().cpu() * batch_size
            if not self._config.batch_gradient_descent:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        if self._config.batch_gradient_descent:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_loss /= num_samples

        return train_loss

    def compute_loss(self,
                     model_outputs: SequenceClassifierOutput,
                     model_inputs: dict):

        if model_inputs['labels'].dtype in [torch.float, torch.float16, torch.float64]:
            logger.error("Undefined!")
            raise NotImplementedError("Functionality has not been defined!")
        elif model_inputs['labels'].dtype in [torch.int, torch.int8, torch.int16, torch.int64]:
            loss = model_outputs.loss
        else:
            logger.error("Unknown label type!")
            raise TypeError('Unknown label type!')
        return loss

    def evaluate(self, dataset: BertClassificationDataset):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._config.device)
        self._model.eval()

        pred_lbs = list()
        with torch.no_grad():
            for inputs in data_loader:
                # get data
                if 'labels' in inputs.keys():
                    inputs.pop('labels')
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._config.device)

                outputs: SequenceClassifierOutput = self._model(**inputs)
                logits = outputs.logits

                pred_prob_batch = F.softmax(logits, dim=-1).detach().to('cpu').numpy()
                pred_lb_batch = pred_prob_batch.argmax(axis=-1).tolist()
                pred_lbs += pred_lb_batch

        true_lbs = np.asarray(dataset.lbs)
        pred_lbs = np.asarray(pred_lbs)

        n_tp = np.sum((pred_lbs == 1) & (true_lbs == 1)) + 1E-9
        n_fp = np.sum((pred_lbs == 1) & (true_lbs == 0)) + 1E-3
        n_fn = np.sum((pred_lbs == 0) & (true_lbs == 1)) + 1E-3

        metric_values = OrderedDict()
        metric_values['accuracy'] = (pred_lbs == true_lbs).sum() / len(pred_lbs)
        metric_values['precision'] = precision = n_tp / (n_tp + n_fp)
        metric_values['recall'] = recall = n_tp / (n_tp + n_fn)
        metric_values['f1'] = 2 * precision * recall / (precision + recall)

        return metric_values

    def predict(self, dataset: BertClassificationDataset):
        data_loader = self.get_dataloader(dataset)
        self._model.to(self._config.device)
        self._model.eval()

        pred_lbs = list()
        pred_probs = list()
        with torch.no_grad():
            for inputs in data_loader:
                # get data
                if 'labels' in inputs:
                    inputs.pop('labels')
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._config.device)

                outputs: SequenceClassifierOutput = self._model(**inputs)
                logits = outputs.logits

                pred_prob_batch = F.softmax(logits, dim=-1).detach().to('cpu').numpy()
                pred_lb_batch = pred_prob_batch.argmax(axis=-1).tolist()
                pred_probs += pred_prob_batch.tolist()
                pred_lbs += pred_lb_batch

        return pred_lbs, pred_probs

    def test(self):
        self._model.to(self._config.device)
        test_results = self.evaluate(self._test_dataset)
        return test_results

    def get_dataloader(self, dataset: BertClassificationDataset, shuffle: Optional[bool] = False):
        if dataset:
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self._config.batch_size,
                collate_fn=self._collate_fn,
                shuffle=shuffle,
                drop_last=False
            )
            return data_loader
        else:
            logger.error('Dataset is not defined!')
            raise ValueError("Dataset is not defined!")

    def save(self, output_dir: Optional[str] = None,
             save_optimizer_and_scheduler: Optional[bool] = False):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory
        save_optimizer_and_scheduler: whether to save optimizer and scheduler

        Returns
        -------
        None
        """
        output_dir = output_dir if output_dir is not None else self._config.output_dir
        logger.info(f"Saving model to {output_dir}")
        self._model.save_pretrained(save_directory=output_dir)
        self._tokenizer.save_pretrained(save_directory=output_dir)
        # Good practice: save your training arguments together with the trained model
        self._config.save(output_dir)
        # save trainer parameters
        if save_optimizer_and_scheduler:
            logger.info("Saving optimizer and scheduler")
            torch.save(self._optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self._lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def load(self, input_dir: Optional[str] = None, load_optimizer_and_scheduler: Optional[bool] = False):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_optimizer_and_scheduler: whether load other trainer parameters

        Returns
        -------
        self
        """
        input_dir = input_dir if input_dir is not None else self._config.output_dir
        if self._model is not None:
            logger.warning(f"The original model {type(self._model)} in {type(self)} is not None. "
                           f"It will be overwritten by the loaded model!")
        logger.info(f"Loading model from {input_dir}")
        self._model = AutoModelForSequenceClassification.from_pretrained(input_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(input_dir)
        if load_optimizer_and_scheduler:
            logger.info("Loading optimizer and scheduler")
            if self._optimizer is None:
                self.set_optimizer_scheduler()
            if os.path.isfile(os.path.join(input_dir, "optimizer.pt")):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, "optimizer.pt"), map_location=self._config.device)
                )
            else:
                logger.warning("Optimizer file does not exist!")
            if os.path.isfile(os.path.join(input_dir, "scheduler.pt")):
                self._lr_scheduler.load_state_dict(torch.load(os.path.join(input_dir, "scheduler.pt")))
            else:
                logger.warning("Learning rate scheduler file does not exist!")
        return self
