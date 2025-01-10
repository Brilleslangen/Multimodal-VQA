from transformers import pipeline
import pandas as pd
from helpers import select_device, load_dataset

tokenizer_config = {'padding': True, 'truncation': True, 'max_length': 512}


def predict_dataset(model: str, test_df):
    classifier = pipeline('text-classification', model=model, device=select_device())
    texts = test_df['text'].tolist()
    predictions = classifier(texts, **tokenizer_config)
    predict_column = [prediction['label'][-1] for prediction in predictions]
    test_df['prediction'] = predict_column

    return test_df


def produce_test_result_matrix(dataset_name, classifier_name, output_name):
    dataset, _ = load_dataset(csv_path=f"../../dataset/{dataset_name}")
    test_df = dataset['test'].to_pandas()

    pred_df = predict_dataset(model=f"{classifier_name}", test_df=test_df)
    pred_df.to_csv(f"../../inference_results/{output_name}")
    return pd.read_csv(f"../../inference_results/{output_name}")
