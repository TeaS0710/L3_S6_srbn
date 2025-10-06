# script_stats.py
import pandas as pd

def compute_stats(json_files, output_csv):
    records = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            tokens = doc["tokens"]
            nb_tokens = len(tokens)
            nb_types = len(set(tokens))
            prop_lemmas = nb_types / nb_tokens if nb_tokens else 0.0
            records.append({
                "filename": doc["filename"],
                "lang": doc["lang"],
                "nb_tokens": nb_tokens,
                "nb_types": nb_types,
                "prop_lemmas": prop_lemmas,
                "prop_ne": doc.get("prop_ne", 0.0)
            })
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8")
