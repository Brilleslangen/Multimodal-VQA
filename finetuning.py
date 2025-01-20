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
    TrainingArguments)

from peft import get_peft_model, LoraConfig
from helpers import load_and_preprocess_dataset, select_device, extract_last_eos_group
from models import PaliGemmaForClassification


class FineTuner:
    def __init__(self, model_id: str, processor_id, classification: bool, freeze_vision: bool, lora: bool,
                 dataset_id: str,
                 test_size: float | int,
                 batch_size: int,
                 image_size,
                 output_folder: str,
                 output_name: str,
                 num_epochs: int = 5,
                 wand_logging: bool = True,
                 eval_steps: int = 50,
                 device=select_device()):
        # Runtime constants
        self.classification_mode = classification
        self.batch_size = batch_size if classification else batch_size
        self.num_epochs = num_epochs
        self.seed = 42
        self.wandb_logging = wand_logging
        self.eval_steps = eval_steps
        self.output_folder = output_folder
        self.output_name = output_name
        self.device = device
        self.sep_token = '\n<separator>\n'

        # Dataset and model
        self.dataset = load_and_preprocess_dataset(dataset_id, classification, self.sep_token, test_size, image_size)
        self.metric_names = ('accuracy',)  # 'recall', 'precision', 'f1'

        # Tokenizer, model and trainer
        self.model = self.init_model(model_id, freeze_vision=freeze_vision, lora=lora)
        self.processor = PaliGemmaProcessor.from_pretrained(processor_id)
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

    def preprocess_logits_for_metrics(self, logits, labels):
        logits = logits if self.classification_mode else logits[0]
        pred_ids = torch.argmax(logits, dim=-1)

        return pred_ids, labels

    def compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
        label_ids = eval_pred.label_ids
        pred_ids = eval_pred.predictions[0]

        print("compute: ", pred_ids, label_ids)

        if self.classification_mode:
            accuracy = evaluate.load('accuracy').compute(predictions=pred_ids, references=label_ids)
        else:
            # Generative accuracy
            pred_ids = np.where(pred_ids != -100, pred_ids, self.processor.tokenizer.pad_token_id)
            predictions = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
            predictions = [extract_last_eos_group(p) for p in predictions]

            label_ids = np.where(label_ids != -100, label_ids, self.processor.tokenizer.pad_token_id)
            labels = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            for prediction, label in zip(predictions, labels):
                print(f"Generation: {prediction} \nAnswer {label}")
            correct = sum([p == l for p, l in zip(predictions, labels)])
            accuracy = correct / len(predictions)

        return {"accuracy": accuracy}

    def collate_fn(self, batch):
        if self.classification_mode:
            # According to https://ai.google.dev/gemma/docs/agile_classifiers#text_preprocessing_and_separator_tokens
            unfolded_batch = {'question_option_pair': [], 'image': [], 'answer': []}

            # 4 New rows for each question
            for row in batch:
                unfolded_batch['answer'].append(row['answer'])
                for pair in row['question_option_pairs']:
                    unfolded_batch['question_option_pair'].append(pair)
                    unfolded_batch['image'].append(row['image'])

            batch = unfolded_batch
            inputs = self.processor(text=batch['question_option_pair'], images=batch['image'], return_tensors="pt",
                                    padding="longest")
            inputs['labels'] = torch.tensor(batch['answer'])
        else:
            questions = [row['question'] for row in batch]
            images = [row['image'] for row in batch]
            answers = [row["answer"] for row in batch]

            inputs = self.processor(text=questions, images=images, suffix=answers, return_tensors="pt",
                                    padding="longest")

        inputs = inputs.to(self.model.dtype).to(self.device)

        return inputs

    def init_trainer(self):
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_folder, self.output_name),
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
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
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

        if not self.classification_mode:
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        else:
            model = PaliGemmaForClassification(model_id, num_labels=4, torch_dtype=torch.bfloat16)

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
