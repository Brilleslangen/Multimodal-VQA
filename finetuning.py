import os
import evaluate
import numpy as np
import wandb
import torch
from transformers import (
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)

from helpers import select_device, extract_last_eos_group, Mode, CosineIndexer, gen_logits_to_indice
from models import init_model
from data_processing import collate_fn, load_and_preprocess_dataset


class FineTuner:
    MODES = [Mode.COND_GEN, Mode.MULTI_CLASS, Mode.SWAG]
    SEP_TOKEN = '\n<separator>\n'

    def __init__(self, model_id: str, processor_id, mode: Mode, attention_pooling: bool, freeze_vision: bool,
                 lora: bool,
                 dataset_id: str,
                 test_size: float | int,
                 batch_size: int,
                 image_size,
                 output_folder: str,
                 output_name: str,
                 num_epochs: float = 5,
                 wand_logging: bool = True,
                 eval_steps: int = 50,
                 quantize: bool = False,
                 device=select_device()):
        # Runtime constants
        self.mode = mode
        self.classification = mode != Mode.COND_GEN
        self.attention_pooling = attention_pooling
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = 42
        self.wandb_logging = wand_logging
        self.eval_steps = eval_steps
        self.output_folder = output_folder
        self.output_name = output_name
        self.device = device
        # According to https://ai.google.dev/gemma/docs/agile_classifiers#text_preprocessing_and_separator_tokens

        # Dataset and model
        self.dataset = load_and_preprocess_dataset(dataset_id, mode, FineTuner.SEP_TOKEN, 'train', test_size,
                                                   image_size)
        self.metric_names = ('accuracy',)  # 'recall', 'precision', 'f1'

        # Tokenizer, model and trainer
        self.model, self.processor, self.parameter_config = init_model(model_id, processor_id, mode=mode,
                                                                       freeze_vision=freeze_vision,
                                                                       lora=lora,
                                                                       quantize=quantize,
                                                                       attention_pooling=attention_pooling,
                                                                       device=device)
        self.trainer = self.init_trainer()

        # Initialize training logger
        if self.wandb_logging:
            self.wandb = wandb.init(project="diagram-vqa",
                                    config={
                                        'base_model': model_id,
                                        'dataset': self.dataset['train'].config_name,
                                        'train_dataset_size': len(self.dataset['train']),
                                        'eval_dataset_size': len(self.dataset['test']),
                                    })
        else:
            os.environ["WANDB_MODE"] = "disabled"

    def preprocess_logits_for_metrics(self, logits, labels):
        logits = logits if self.classification else logits[0]
        pred_ids = torch.argmax(logits, dim=-1)

        return pred_ids, labels

    def compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
        pred_ids = eval_pred.predictions[0]

        if self.mode == Mode.COND_GEN:
            pred_ids = gen_logits_to_indice(pred_ids, self.processor, self.dataset['test']['options'])

        accuracy = evaluate.load('accuracy').compute(predictions=pred_ids, references=self.dataset['test']['answer'])
        return accuracy

    def init_trainer(self):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_folder, self.output_name),
            include_for_metrics=["inputs"],
            eval_strategy='steps',
            eval_steps=self.eval_steps,
            save_strategy='steps',
            save_steps=500,
            optim='adamw_torch',
            bf16=True,
            num_train_epochs=self.num_epochs,
            auto_find_batch_size=True,
            load_best_model_at_end=False,
            remove_unused_columns=False,
            per_device_train_batch_size=self.batch_size,  # Reduce to lower memory requirements
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=2,
            # learning_rate=5e-5,
            # weight_decay=1e-6,
            # adam_beta2=0.999,
            logging_steps=200,
            save_total_limit=1,
            dataloader_pin_memory=False,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            data_collator=lambda batch: collate_fn(batch, self.model, self.processor, self.mode, training=True),
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )

    def train(self):
        self.trainer.train()
        folder = os.path.join(self.output_folder, self.output_name)
        self.trainer.save_model(folder)
        self.parameter_config.save_to_file(folder)

    def evaluate(self):
        return self.trainer.evaluate()
