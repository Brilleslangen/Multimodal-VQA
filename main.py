import os
import time

import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, AutoModel, PaliGemmaProcessor, PaliGemmaPreTrainedModel, \
    PaliGemmaForConditionalGeneration
import torch

from finetuning import FineTuner
from data_processing import load_and_preprocess_dataset, collate_fn
from models import init_model, PaliGemmaForClassification
from helpers import Mode, select_device, CosineIndexer, ParameterConfig, gen_logits_to_indice

image_size = 448
model_id = f'google/paligemma2-10b-pt-{image_size}'
dataset_folder = 'datasets/diagram-vqa/'

model_output_path = 'models-pt/'


def train(model_name_extras="", mode=Mode.COND_GEN, attention_pooling=False, freeze_vision=False, lora=True,
          quantize=False):
    model_name_extras = ("ATT" if attention_pooling else "") + model_name_extras
    _model_name = "PG2" + model_id[17:] + '-' + mode.value + (("-" + model_name_extras) if model_name_extras else "")

    print(f'Train {_model_name}')
    finetuner = FineTuner(model_id=model_id,
                          processor_id=model_id,
                          mode=mode,
                          attention_pooling=attention_pooling,
                          freeze_vision=freeze_vision,
                          lora=lora,
                          quantize=quantize,
                          dataset_id=dataset_folder + 'train',
                          test_size=1,
                          image_size=(image_size, image_size),
                          batch_size=8,
                          output_folder=model_output_path,
                          output_name=_model_name,
                          num_epochs=2,
                          wand_logging=True,
                          eval_steps=0)
    finetuner.train()
    results = finetuner.evaluate()
    print(results, '\n')

    return model_output_path + _model_name


def evaluate(_model_path, split, batch_size=1):
    print(f'Evaluate {_model_path} on {split}')
    device = select_device()

    # Load model & processor
    config = ParameterConfig.load_from_file(_model_path)
    if config.mode == Mode.COND_GEN:
        model = PaliGemmaForConditionalGeneration.from_pretrained(_model_path)
    else:
        model = PaliGemmaForClassification.from_pretrained(_model_path, mode=config.mode, num_labels=4,
                                                           attention_pooling=config.attention_pooling)
    model.to(device)
    processor = PaliGemmaProcessor.from_pretrained(model_id)

    # Prepare dataset
    dataset = load_and_preprocess_dataset(dataset_id=dataset_folder + split, mode=config.mode,
                                          sep_token=FineTuner.SEP_TOKEN, split=split,
                                          image_size=(image_size, image_size), test_size=0).select(range(100))
    image_names = sorted(pd.read_csv(f'datasets/diagram-vqa/{split}-metadata.csv')['file_name'].tolist())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, model, processor, config.mode, training=False))

    # Evaluate
    all_predictions = []
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            outputs = model(**batch)
            predictions = torch.argmax(outputs['logits'], dim=-1).cpu().numpy()

            if config.mode == Mode.COND_GEN:
                predictions = gen_logits_to_indice(predictions, processor, dataset['options'])

            if hasattr(batch['images'], 'filename') and batch['images'].filename:
                # Extract filename using os.path to handle different OS path formats
                batch['file_name'] = os.path.basename(batch['images'].filename)

            predictions = [int(i + 1) for i in predictions]
            print(batch['file_name'], predictions)
            all_predictions.extend(predictions)

    # Save predictions
    output_dir = "evaluations"
    output_file = output_dir + '/' + _model_path.split('/')[-1] + ".csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame({"file_name": image_names,
                  "answer": all_predictions}).to_csv(output_file, index=False)

    print('Evaluation complete and saved to evaluations/' + _model_path.split('/')[-1] + ".csv")


# Conditional generation
# model_path = train("", Mode.COND_GEN, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
evaluate('models-pt/PG2-10b-pt-448-COND_GEN-ATT', 'train')

# Multi-class classification with and without attention pooling
# model_path = train("", Mode.MULTI_CLASS, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
# evaluate(model_path, 'validate')

# model_path = train("", Mode.MULTI_CLASS, attention_pooling=True, freeze_vision=True, lora=True, quantize=False)
# evaluate(model_path, 'validate')

# SWAG with and without attention pooling
# model_path = train("", Mode.SWAG, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
# evaluate(model_path, 'validate')

# model_path = train("", Mode.SWAG, attention_pooling=True, freeze_vision=True, lora=True, quantize=False)
# evaluate(model_path, 'validate')
