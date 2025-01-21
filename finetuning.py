import os
import evaluate
import numpy as np
import wandb
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig)

from peft import get_peft_model, LoraConfig
from helpers import load_and_preprocess_dataset, select_device, extract_last_eos_group, Mode
from models import PaliGemmaForClassification


class FineTuner:
    MODES = [Mode.COND_GEN, Mode.MULTI_CLASS, Mode.SWAG]

    def __init__(self, model_id: str, processor_id, mode: Mode, attention_pooling: bool, freeze_vision: bool, lora: bool,
                 dataset_id: str,
                 test_size: float | int,
                 batch_size: int,
                 image_size,
                 output_folder: str,
                 output_name: str,
                 num_epochs: int = 5,
                 wand_logging: bool = True,
                 eval_steps: int = 50,
                 qlora: bool = False,
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
        self.sep_token = '\n<separator>\n'
        # According to https://ai.google.dev/gemma/docs/agile_classifiers#text_preprocessing_and_separator_tokens

        # Dataset and model
        self.dataset = load_and_preprocess_dataset(dataset_id, mode, self.sep_token, test_size, image_size)
        self.metric_names = ('accuracy',)  # 'recall', 'precision', 'f1'

        # Tokenizer, model and trainer
        self.model = self.init_model(model_id, freeze_vision=freeze_vision, lora=lora)
        self.processor = PaliGemmaProcessor.from_pretrained(processor_id)
        self.trainer = self.init_trainer()

        # Cosine similarity model
        if self.mode == Mode.COND_GEN:
            self.cosine_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.cosine_model.similarity_fn_name = SimilarityFunction.COSINE
        else:
            self.cosine_model = None

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
        logits = logits if self.classification else logits[0]
        pred_ids = torch.argmax(logits, dim=-1)

        return pred_ids.to(self.model.dtype).to(self.device), labels.to(self.model.dtype).to(self.device)

    def compute_metrics(self, eval_pred):
        """Function for computing evaluation metrics"""
        pred_ids = eval_pred.predictions[0]

        if self.mode == Mode.COND_GEN:
            # Select choice with the highest cosine similarity
            pred_ids = np.where(pred_ids != -100, pred_ids, self.processor.tokenizer.pad_token_id)
            predictions = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
            predictions = [extract_last_eos_group(p) for p in predictions]
            pred_ids = self.cosine_sim_to_label_indice(predictions, self.dataset['test']['options'])

        accuracy = evaluate.load('accuracy').compute(predictions=pred_ids, references=self.dataset['test']['answer'])
        return {"accuracy": accuracy}

    def collate_fn(self, batch):
        if self.mode == Mode.SWAG:
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
        elif self.mode == Mode.MULTI_CLASS:
            questions = [row['question'] for row in batch]
            images = [row['image'] for row in batch]
            answers = [row["answer"] for row in batch]

            inputs = self.processor(text=questions, images=images, return_tensors="pt",
                                    padding="longest")
            inputs['labels'] = torch.tensor(answers)
        else:
            questions = [row['question'] for row in batch]
            images = [row['image'] for row in batch]
            answers = [row["text_answer"] for row in batch]

            inputs = self.processor(text=questions, images=images, suffix=answers, return_tensors="pt",
                                    padding="longest")

        inputs = inputs.to(self.model.dtype).to(self.device)
        print(inputs['input_ids'].dtype, inputs['attention_mask'].dtype, inputs['labels'].dtype)

        return inputs

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
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics
        )

    def init_model(self, model_id, freeze_vision=False, lora=True, qlora=False):
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

        if qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type=torch.bfloat16)

        if self.classification:
            model = PaliGemmaForClassification.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                               attn_implementation='eager',
                                                               quantization_config=bnb_config if qlora else None,
                                                               swag_mode=self.mode == Mode.SWAG, num_labels=4,
                                                               attention_pooling=self.attention_pooling)
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                                      attn_implementation='eager',
                                                                      quantization_config=bnb_config if qlora else None)

        model.config.keys_to_ignore_at_inference = ["past_key_values"]

        if lora:
            model = get_peft_model(model, lora_config)

        if freeze_vision:
            for param in model.vision_tower.parameters():
                param.requires_grad = False

            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False

        return model.to(self.device)

    def cosine_sim_to_label_indice(self, predictions, options):
        """
        Given a list of string predictions and a list of string options (both in English and Japanese),
        this function returns the index of the option that has the highest cosine similarity with the predicted text.
        """

        indices = []

        for pred, opts in zip(predictions, options):
            # Encode prediction and all options
            text_list = [pred] + opts  # Combine prediction and options
            embeddings = self.cosine_model.encode(text_list, convert_to_numpy=True)

            # Compute cosine similarities between prediction and options
            pred_embedding = embeddings[0].reshape(1, -1)
            option_embeddings = embeddings[1:]

            similarities = self.cosine_model.similarity(pred_embedding, option_embeddings)
            best_index = np.argmax(similarities)

            indices.append(best_index)

        return indices

    def train(self):
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_folder, self.output_name))

    def evaluate(self):
        return self.trainer.evaluate()
