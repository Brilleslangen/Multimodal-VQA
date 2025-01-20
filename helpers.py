import re
from enum import Enum

import torch
from datasets import Dataset, load_dataset
from datasets import DatasetDict


class Mode(Enum):
    COND_GEN = 'Conditional generation'
    MULTI_CLASS = 'Classical multi-class classification'
    SWAG = 'SWAG-based multi-class classification'


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


def load_and_preprocess_dataset(dataset_id, mode: Mode, sep_token, test_size=0.05, image_size=(224, 224)) -> (
        DatasetDict, int):
    def resize_images(batch):
        batch['image'] = [image.resize(image_size) for image in batch['image']]
        return batch

    def insert_image(batch):
        batch['question'] = [f"<image> {question}" for question in batch['question']]
        return batch

    def map_to_text_answer(batch):
        batch['text_answer'] = [batch[f'option{ans}'][i] for i, ans in enumerate(batch['answer'])]
        return batch

    def map_to_label_indices(batch):
        batch['answer'] = [ans - 1 for ans in batch['answer']]
        return batch

    def preprocess_for_conditional_gen(batch):
        """
        Modify the 'question' field to include all four options concatenated with a separator.
        E.g., "Question text Options: Option1 | Option2 | Option3 | Option4"
        """
        batch['question'] = [question + sep_token
                             + 'Options: ' + ' | '.join([batch[f'option{i}'][idx] for i in range(1, 5)]) + sep_token
                             + 'Answer:'
                             for idx, question in enumerate(batch['question'])]
        batch['options'] = [[f"{batch[f'option{i}'][idx]}" for i in range(1, 5)]
                            for idx in range(len(batch['question']))]

        return batch

    def preprocess_for_SWAG(batch):
        batch['question_option_pairs'] = [[f"{question}{sep_token}Hypothesis: {batch[f'option{i}'][idx]}"
                                           for i in range(1, 5)] for idx, question in enumerate(batch['question'])]
        batch.pop('question', None)
        return batch

    def preprocess_for_multi_class(batch):
        enumerators = ['A', 'B', 'C', 'D']
        batch['question'] = [question + sep_token
                             + 'Options: ' + ' | '.join([enumerators[i - 1] + ") " + batch[f'option{i}'][idx]
                                                         for i in range(1, 5)]) + sep_token
                             + 'Answer:'
                             for idx, question in enumerate(batch['question'])]

        return batch

    dataset = load_dataset(dataset_id, split='train')
    dataset = dataset.map(resize_images, batched=True)
    dataset = dataset.map(insert_image, batched=True)

    if mode == Mode.MULTI_CLASS:
        dataset = dataset.map(map_to_label_indices, batched=True)
        dataset = dataset.map(preprocess_for_multi_class, batched=True)

    elif mode == Mode.SWAG:
        dataset = dataset.map(map_to_label_indices, batched=True)
        dataset = dataset.map(preprocess_for_SWAG, batched=True)
    else:
        dataset = dataset.map(preprocess_for_conditional_gen, batched=True)
        dataset = dataset.map(map_to_text_answer, batched=True)

    dataset = dataset.remove_columns([f'option{i}' for i in range(1, 5)])
    train_test_split = dataset.train_test_split(seed=42, test_size=test_size)

    return train_test_split
