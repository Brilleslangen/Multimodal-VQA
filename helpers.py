import json
import re
from enum import Enum

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SimilarityFunction


class Mode(Enum):
    COND_GEN = 'COND_GEN'
    MULTI_CLASS = 'MULTI_CLASS'
    SWAG = 'SWAG'

    def __str__(self):
        return self.value


class ParameterConfig:
    CONFIG_NAME = 'parameter_config.json'

    def __init__(self, mode: Mode, attention_pooling: bool, freeze_vision: bool = False, lora: bool = True,
                 quantize: bool = False):
        self.mode = mode
        self.attention_pooling = attention_pooling
        self.freeze_vision = freeze_vision
        self.lora = lora
        self.quantize = quantize

    def to_dict(self):
        """Convert the config to a dictionary format for saving to JSON."""
        return {
            "mode": self.mode.value,  # Convert enum to string
            "attention_pooling": self.attention_pooling,
            "freeze_vision": self.freeze_vision,
            "lora": self.lora,
            "quantize": self.quantize
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Load the config from a dictionary."""
        return cls(
            mode=Mode(config_dict["mode"]),  # Convert string back to enum
            attention_pooling=config_dict["attention_pooling"],
            freeze_vision=config_dict["freeze_vision"],
            lora=config_dict["lora"],
            quantize=config_dict["quantize"]
        )

    def save_to_file(self, model_path):
        """Save config to a JSON file."""
        with open(model_path + "/" + ParameterConfig.CONFIG_NAME, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, model_path):
        """Load config from a JSON file."""
        with open(model_path + "/" + ParameterConfig.CONFIG_NAME, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self):
        return f"ClassificationConfig(mode={self.mode}, attention_pooling={self.attention_pooling}, freeze_vision={self.freeze_vision}, lora={self.lora}, quantize={self.quantize})"


class CosineIndexer:
    def __init__(self):
        self.cosine_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.cosine_model.similarity_fn_name = SimilarityFunction.COSINE

    def convert(self, predictions, options):
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


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def extract_last_eos_group(text):
    if '<eos>' in text:
        text = re.sub(r'(<eos>\s*)+', '<eos>', text)
        matches = re.findall(r'(.*?)<eos>', text)
        return matches[-1].strip() if matches else text.strip()
    return text.strip()


def gen_logits_to_indice(pred_ids, processor, options):
    # Select choice with the highest cosine similarity
    pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
    predictions = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    predictions = [extract_last_eos_group(p) for p in predictions]
    pred_ids = CosineIndexer().convert(predictions, options)
    print(pred_ids, predictions, options)

    return pred_ids

