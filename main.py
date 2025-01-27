import os
import time

import numpy as np
import evaluate as eval
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline, AutoModel, PaliGemmaProcessor, PaliGemmaPreTrainedModel, \
    PaliGemmaForConditionalGeneration
import torch

from finetuning import FineTuner
from data_processing import load_and_preprocess_dataset, collate_fn, metadata_csv_to_jsonl
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
                          batch_size=1,
                          output_folder=model_output_path,
                          output_name=_model_name,
                          num_epochs=2,
                          wand_logging=True,
                          eval_steps=1)
    finetuner.train()
    results = finetuner.evaluate()
    print(results, '\n')

    return model_output_path + _model_name


def evaluate(_model_path, split, batch_size=1, labeled=False):
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
                                          image_size=(image_size, image_size), test_size=0)

    dataset = dataset.select(range(100))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, model, processor, config.mode, training=False,
                                                            include_image_name=True, eval_debug=labeled))

    # Evaluate
    image_names = []
    all_predictions = []
    answers = []
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            options = batch.pop('options')
            image_batch_names = batch.pop('image_names')

            if labeled:
                answers.extend(batch.pop('answers'))

            outputs = model.generate(**batch)

            if config.mode == Mode.COND_GEN:
                predictions = gen_logits_to_indice(outputs.cpu().numpy(), processor, options)

            predictions = [int(i + 1) for i in predictions]
            answers = [int(i + 1) for i in answers]

            image_names.extend(image_batch_names)
            all_predictions.extend(predictions)

    # Test
    if labeled:
        acc = eval.load('accuracy').compute(predictions=all_predictions, references=answers)
        print(acc)

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
evaluate('models-pt/PG2-10b-pt-448-COND_GEN-ATT', 'train', labeled=True)

# metadata_csv_to_jsonl('datasets/diagram-vqa/train-metadata.csv', 'datasets/diagram-vqa/train/metadata.jsonl')
# metadata_csv_to_jsonl('datasets/diagram-vqa/validate-metadata.csv', 'datasets/diagram-vqa/validate/metadata.jsonl')
# metadata_csv_to_jsonl('datasets/diagram-vqa/test-metadata.csv', 'datasets/diagram-vqa/test/metadata.jsonl')

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
