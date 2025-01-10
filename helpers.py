import torch
from datasets import Dataset, load_dataset as load_ds
from datasets import DatasetDict


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_dataset(dataset_id, test_size=0.001, image_size=(224, 224)) -> (DatasetDict, int):
    """
    Loads a dataset from a CSV file, preprocesses it, and splits it into training and test sets.

    Args:
        image_size: Tuple of image dims
        self (str): The file path to the CSV file containing the dataset.
        dataset_id (str): The dataset id.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' datasets.
    """
    # Load data from CSV

    def resize_images(batch):
        batch['image'] = [image.resize(image_size) for image in batch['image']]
        return batch

    dataset: Dataset = load_ds(dataset_id, split='train')
    dataset = dataset.map(resize_images, batched=True)
    train_test_split = dataset.train_test_split(seed=42, test_size=test_size)

    return train_test_split
