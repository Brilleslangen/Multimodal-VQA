import pandas as pd
import torch
from datasets import DatasetDict, load_dataset

from helpers import Mode


def metadata_csv_to_jsonl(in_path, out_path):
    metadata = pd.read_csv(in_path)
    metadata.to_json(out_path, force_ascii=False, orient="records", lines=True)


def load_and_preprocess_dataset(dataset_id, mode: Mode, sep_token, split: str, test_size=0.05,
                                image_size=(224, 224)) -> (
        DatasetDict, int):
    def resize_images(batch):
        batch['images'] = [image.resize(image_size) for image in batch['image']]
        batch.pop('image')
        return batch

    def insert_image(batch):
        batch['text'] = [f"<image> {question}" for question in batch['question']]
        batch.pop('question')
        return batch

    def map_to_text_answer(batch):
        if split == 'train':
            batch['text_answer'] = [batch[f'option{ans + 1}'][i] for i, ans in enumerate(batch['answer'])]
        return batch

    def map_to_label_indices(batch):
        if split == 'train':
            batch['answer'] = [ans - 1 for ans in batch['answer']]
        return batch

    def preprocess_for_conditional_gen(batch):
        """
        Modify the 'question' field to include all four options concatenated with a separator.
        E.g., "Question text Options: Option1 | Option2 | Option3 | Option4"
        """
        batch['text'] = [question + sep_token
                         + 'Options: ' + ' | '.join([batch[f'option{i}'][idx] for i in range(1, 5)]) + sep_token
                         + 'Answer:'
                         for idx, question in enumerate(batch['text'])]
        batch['options'] = [[f"{batch[f'option{i}'][idx]}" for i in range(1, 5)]
                            for idx in range(len(batch['text']))]

        return batch

    def preprocess_for_SWAG(batch):
        batch['text'] = [[f"{question}{sep_token}Hypothesis: {batch[f'option{i}'][idx]}"
                          for i in range(1, 5)] for idx, question in enumerate(batch['text'])]
        return batch

    def preprocess_for_multi_class(batch):
        enumerators = ['A', 'B', 'C', 'D']
        batch['text'] = [question + sep_token
                         + 'Options: ' + ' | '.join([enumerators[i - 1] + ") " + batch[f'option{i}'][idx]
                                                     for i in range(1, 5)]) + sep_token
                         + 'Answer:'
                         for idx, question in enumerate(batch['text'])]

        return batch

    dataset = load_dataset(dataset_id, split='train')
    dataset = dataset.map(resize_images, batched=True)
    dataset = dataset.map(insert_image, batched=True)
    dataset = dataset.map(map_to_label_indices, batched=True)

    if mode == Mode.MULTI_CLASS:
        dataset = dataset.map(preprocess_for_multi_class, batched=True)

    elif mode == Mode.SWAG:
        dataset = dataset.map(preprocess_for_SWAG, batched=True)
    else:
        dataset = dataset.map(preprocess_for_conditional_gen, batched=True)
        dataset = dataset.map(map_to_text_answer, batched=True)

    dataset = dataset.remove_columns([f'option{i}' for i in range(1, 5)])

    if split == 'train':
        train_test_split = dataset.train_test_split(seed=42, test_size=test_size)
        return train_test_split

    return dataset


def collate_fn(batch, model, processor, mode, training: bool = True):
    if mode == Mode.SWAG:
        unfolded_batch = {'question_option_pair': [], 'images': [], 'answer': []}

        # 4 New rows for each question
        for row in batch:
            if training:
                unfolded_batch['answer'].append(row['answer'])

            for pair in row['text']:
                unfolded_batch['question_option_pair'].append(pair)
                unfolded_batch['images'].append(row['images'])

        batch = unfolded_batch
        inputs = processor(text=batch['question_option_pair'], images=batch['images'], return_tensors="pt",
                           padding="longest")
        if training:
            inputs['labels'] = torch.tensor(batch['answer'])

    elif mode == Mode.MULTI_CLASS:
        questions_options = [row['text'] for row in batch]
        images = [row['images'] for row in batch]

        inputs = processor(text=questions_options, images=images, return_tensors="pt",
                           padding="longest")
        if training:
            inputs['labels'] = torch.tensor([row["answer"] for row in batch])

    elif mode == Mode.COND_GEN:
        questions = [row['text'] for row in batch]
        images = [row['images'] for row in batch]

        if training:
            answers = [row["text_answer"] for row in batch]
            inputs = processor(text=questions, images=images, suffix=answers, return_tensors="pt",
                           padding="longest")
        else:
            inputs = processor(text=questions, images=images, return_tensors="pt", padding="longest")
    else:
        raise ValueError(f"Unknown mode {mode}")

    inputs = inputs.to(model.dtype).to(model.device)

    return inputs
