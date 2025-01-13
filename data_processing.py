import pandas as pd


def metadata_csv_to_jsonl(in_path, out_path):
    metadata = pd.read_csv(in_path)

    metadata['question'] = metadata.apply(
        lambda row: row['question'] + ' Options: ' + ' | '.join(
            [f"{i}) " + row[f'option{i}'] for i in range(1, 5)]
        ), axis=1
    )

    metadata['answer'] = metadata.apply(lambda row: row['answer'], axis=1)
    metadata.to_json(out_path, force_ascii=False, orient="records", lines=True)


metadata_csv_to_jsonl("../datasets/diagram-vqa/metadata.csv", "../datasets/diagram-vqa/train/metadata.jsonl")
