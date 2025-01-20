import pandas as pd


def metadata_csv_to_jsonl(in_path, out_path):
    metadata = pd.read_csv(in_path)
    metadata.to_json(out_path, force_ascii=False, orient="records", lines=True)


metadata_csv_to_jsonl("../datasets/diagram-vqa/metadata.csv", "../datasets/diagram-vqa/train/metadata.jsonl")
