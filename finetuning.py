import os
import re

import evaluate
import numpy as np
import wandb
import torch
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding)

from peft import get_peft_model, LoraConfig
from helpers import load_dataset, select_device


class FineTuner:
    def __init__(self, model_id: str, freeze_vision: bool, lora: bool, dataset_id: str, image_size,
                 output_folder: str,
                 output_name: str,
                 num_epochs: int = 5,
                 wand_logging: bool = True,
                 eval_steps: int = 50,
                 device=select_device()):
        # Runtime constants
        self.num_epochs = num_epochs
        self.seed = 42
        self.wandb_logging = wand_logging
        self.eval_steps = eval_steps
        self.output_folder = output_folder
        self.output_name = output_name
        self.device = device

        # Dataset and model
        self.dataset = load_dataset(dataset_id, image_size=image_size)
        self.metric_names = ('accuracy',)  # 'recall', 'precision', 'f1'

        # Tokenizer, model and trainer
        self.model = self.init_model(model_id, freeze_vision=freeze_vision, lora=lora)
        self.processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.collator = DataCollatorWithPadding(self.processor.tokenizer, padding=True)
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
            os.environ["WANDB_DISABLED"] = "true"

    def compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
        predictions, labels = eval_pred
        predictions = self.processor.tokenizer.batch_decode(
            np.argmax(predictions[0], axis=-1), skip_special_tokens=False)

        # Cut off from first eos token with regex
        predictions = [re.match(r'^(.*?)<eos>', p).group(1).strip()
                       if '<eos>' in p else p.strip() for p in predictions]

        # Convert label input_ids to text
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        labels = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

        correct = sum([p == l for p, l in zip(predictions, labels)])
        accuracy = correct / len(predictions)

        print(f"Predictions: {predictions}")
        print(f"Labels: {labels}")

        return {"accuracy": accuracy}

    def collate_fn(self, examples):
        texts = ["<image>" + example["question"] for example in examples]
        labels = [example["answer"] for example in examples]
        images = [example["image"] for example in examples]
        inputs = self.processor(text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest")
        inputs = inputs.to(self.model.dtype).to(self.device)

        return inputs

    def init_trainer(self):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_folder, self.output_name),
            eval_strategy='steps',
            eval_steps=self.eval_steps,
            save_strategy='steps',
            save_steps=500,
            optim='adamw_hf',
            bf16=True,
            num_train_epochs=self.num_epochs,
            auto_find_batch_size=True,
            load_best_model_at_end=False,
            remove_unused_columns=False,
            per_device_train_batch_size=2,  # Reduce to lower memory requirements
            per_device_eval_batch_size=2,
            warmup_steps=2,
            learning_rate=5e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=200,
            save_total_limit=1,
            dataloader_pin_memory=False,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
        )

    def init_model(self, model_id, freeze_vision=False, lora=True):
        lora_config = LoraConfig(
            r=8,
            target_modules=[
                "q_proj",
                "o_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM",
        )

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16)
        model.config.keys_to_ignore_at_inference = ["past_key_values"]

        if lora:
            model = get_peft_model(model, lora_config)

        if freeze_vision:
            for param in model.vision_tower.parameters():
                param.requires_grad = False

            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False

        return model.to(self.device)

    def train(self):
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_folder, self.output_name))

    def evaluate(self):
        return self.trainer.evaluate()
