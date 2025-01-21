import pandas as pd
from transformers import pipeline
import torch

from finetuning import FineTuner
from models import CosineIndexer
from helpers import Mode, load_and_preprocess_dataset, select_device

image_size = 224
model_id = f'google/paligemma2-3b-pt-{image_size}'
dataset_id = 'datasets/diagram-vqa/validate'

model_output_path = 'model_checkpoints/'


def train():
    print('Train')
    finetuner = FineTuner(model_id=model_id,
                          processor_id=model_id,
                          mode=Mode.SWAG,
                          attention_pooling=False,
                          freeze_vision=True,
                          lora=True,
                          qlora=False,
                          dataset_id=dataset_id,
                          test_size=1,
                          image_size=(image_size, image_size),
                          batch_size=1,
                          output_folder="../model_checkpoints",
                          output_name='Paligemma-VQA',
                          num_epochs=0.005,
                          wand_logging=True,
                          eval_steps=0)
    finetuner.train()
    results = finetuner.evaluate()
    print(results)


def evaluate(saved_model_name, split, output_name):
    pipe = pipeline(model=saved_model_name, device=select_device())
    dataset = load_and_preprocess_dataset(dataset_id, Mode.SWAG, FineTuner.SEP_TOKEN, split, image_size)
    dataset = dataset.rename_columns({'question_option_pairs': 'text', 'image': 'images'})

    # reduce dataset to 5 samples
    dataset = dataset.select(range(2))

    with torch.no_grad():
        predictions = pipe(dataset, batch_size=1)

    if pipe.model.config.mode == Mode.COND_GEN:
        predictions = CosineIndexer().convert(predictions, dataset['options'])

    print(predictions)
    predictions.to_pandas().to_csv(model_output_path + output_name)


# train()

evaluate(model_output_path + 'Paligemma-VQA', 'validate', 'Paligemma-VQA.csv')