import pandas as pd
import json

def normalize_sample(sample):

    return {
        "doc_id": sample["doc_id"],
        "id": sample["doc"]["id"],
        "title": sample["doc"]["title"],
        "context": sample["doc"]["context"],
        "question": sample["doc"]["question"],
        "answers": sample["doc"]["answers"]["text"],
        "target": sample["target"],
        "prediction": sample["resps"][0][0],
    }

def load_predictions(file_path: str) -> pd.DataFrame:
    with open(file_path) as f:
        data = json.load(f)

    samples = data["samples"]["squadv2"]

    return pd.DataFrame.from_records(map(normalize_sample, samples))