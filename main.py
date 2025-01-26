import os
import time

import numpy as np
import pandas as pd
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
train_dataset_id = 'datasets/diagram-vqa/train'
validate_dataset_id = 'datasets/diagram-vqa/validate'

model_output_path = 'models-pt/'


def train(model_name_extras="", mode=Mode.COND_GEN, attention_pooling=False, freeze_vision=False, lora=True,
          quantize=False):
    model_name_extras = "ATT" + model_name_extras
    _model_name = "PG2" + model_id[17:] + '-' + mode.value + (("-" + model_name_extras) if model_name_extras else "")

    print(f'Train {_model_name}')
    finetuner = FineTuner(model_id=model_id,
                          processor_id=model_id,
                          mode=mode,
                          attention_pooling=attention_pooling,
                          freeze_vision=freeze_vision,
                          lora=lora,
                          quantize=quantize,
                          dataset_id=train_dataset_id,
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


def evaluate(_model_path, split, batch_size=8):
    print(f'Evaluate {_model_path} on {split}')

    # Load model & processor
    config = ParameterConfig.load_from_file(_model_path)
    if config.mode == Mode.COND_GEN:
        model = PaliGemmaForConditionalGeneration.from_pretrained(_model_path)
    else:
        model = PaliGemmaForClassification.from_pretrained(_model_path, mode=config.mode, num_labels=4,
                                                           attention_pooling=config.attention_pooling)

    processor = PaliGemmaProcessor.from_pretrained(model_id)

    # Prepare dataset
    dataset = load_and_preprocess_dataset(validate_dataset_id, config.mode, FineTuner.SEP_TOKEN, split, image_size)
    image_names = pd.read_csv(f'datasets/diagram-vqa/{split}-metadata.csv')['file_name'].tolist()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, model, processor, config.mode, training=False))

    # Evaluate
    all_predictions = []
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            outputs = model(**batch)
            predictions = torch.argmax(outputs["logits"], dim=-1)
            all_predictions.append(predictions.cpu().numpy()[0])

    # Post-process predictions from sequence logits to indices for conditional generation
    if config.mode == Mode.COND_GEN:
        all_predictions = gen_logits_to_indice(all_predictions[0], processor, dataset['options'])
        all_predictions = [p.numpy().tolist() for p in all_predictions]
    all_predictions = [i + 1 for i in all_predictions]

    # Save predictions
    output_dir = "evaluations"
    output_file = output_dir + '/' +  _model_path.split('/')[-1] + ".csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame({"file_name": image_names,
                  "answer": all_predictions}).to_csv(output_file, index=False)

    print('Evaluation complete and saved to evaluations/' + _model_path.split('/')[-1] + ".csv")


# Conditional generation
#model_path = train("", Mode.COND_GEN, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
#evaluate(model_path, 'validate')

# Multi-class classification with and without attention pooling
#model_path = train("", Mode.MULTI_CLASS, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
#evaluate(model_path, 'validate')

#model_path = train("", Mode.MULTI_CLASS, attention_pooling=True, freeze_vision=True, lora=True, quantize=False)
#evaluate(model_path, 'validate')

# SWAG with and without attention pooling
model_path = train("", Mode.SWAG, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
#evaluate(model_path, 'validate')

#model_path = train("", Mode.SWAG, attention_pooling=True, freeze_vision=True, lora=True, quantize=False)
#evaluate(model_path, 'validate')
