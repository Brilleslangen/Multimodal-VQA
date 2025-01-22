import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import pipeline
import torch

from finetuning import FineTuner
from data_processing import load_and_preprocess_dataset, collate_fn
from models import init_model
from helpers import Mode, select_device, CosineIndexer, ParameterConfig, gen_logits_to_indice

image_size = 224
model_id = f'google/paligemma2-3b-pt-{image_size}'
train_dataset_id = 'datasets/diagram-vqa/train'
validate_dataset_id = 'datasets/diagram-vqa/validate'

model_output_path = 'models/'


def gen_model_name(model_name_extras="", mode=Mode.COND_GEN):
    return "PG2" + model_id[17:] + '-' + mode.value + (("-" + model_name_extras) if model_name_extras else "")


def train(model_name_extras="", mode=Mode.COND_GEN, attention_pooling=False, freeze_vision=False, lora=True,
          quantize=False):
    model_name = gen_model_name(model_name_extras, mode)
    print(f'Train {model_name}')
    finetuner = FineTuner(model_id=model_id,
                          processor_id=model_id,
                          mode=mode,
                          attention_pooling=attention_pooling,
                          freeze_vision=freeze_vision,
                          lora=lora,
                          quantize=quantize,
                          dataset_id=train_dataset_id,
                          test_size=3,
                          image_size=(image_size, image_size),
                          batch_size=1,
                          output_folder=model_output_path,
                          output_name=model_name,
                          num_epochs=0.000125,
                          wand_logging=False,
                          eval_steps=0)
    finetuner.train()
    results = finetuner.evaluate()
    print(results)

    return model_name


def evaluate(_model_name, split, batch_size=1):
    print(f'Evaluate {_model_name} on {split}')

    config = ParameterConfig.load_from_file(model_output_path + _model_name)
    model, processor, config_2 = init_model(model_output_path + _model_name, model_id, config.mode,
                                            config.attention_pooling, config.freeze_vision, config.lora,
                                            config.quantize, select_device())
    model.eval()

    dataset = load_and_preprocess_dataset(validate_dataset_id, config.mode, FineTuner.SEP_TOKEN, split, image_size)

    image_names = pd.read_csv(f'datasets/diagram-vqa/{split}-metadata.csv')['file_name'].tolist()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, model, processor, config.mode, training=False))
    all_predictions = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model.forward(**batch)
            predictions = torch.argmax(outputs["logits"], dim=-1)
            all_predictions.append(predictions.cpu().numpy()[0])

    if config.mode == Mode.COND_GEN:
        all_predictions = gen_logits_to_indice(all_predictions[0], processor, dataset['options'])
        all_predictions = [p.numpy().tolist() for p in all_predictions]

    all_predictions = [i + 1 for i in all_predictions]

    output_dir = "evaluations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pd.DataFrame({"filename": image_names, "answer": all_predictions}).to_csv('evaluations/' + _model_name + ".csv", index=False)


model_name = train("", Mode.SWAG, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
evaluate(model_name, 'validate')

model_name = train("", Mode.MULTI_CLASS, attention_pooling=False, freeze_vision=True, lora=True, quantize=False)
evaluate(model_name, 'validate')

