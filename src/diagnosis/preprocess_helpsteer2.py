# file: src/preprocess_helpsteer2.py
import json
import pandas as pd

def preprocess_helpsteer2(input_path: str, output_csv: str):
    """
    Expands multi-annotator ratings arrays (helpfulness, correctness, etc.)
    into long-form rows: one row per annotator per dimension.
    """
    rows = []
    with open(input_path, "r") as f:
        for pid, line in enumerate(f):
            ex = json.loads(line)
            prompt = ex["prompt"]
            response = ex["response"]
            helpfulness = ex["helpfulness"]
            for annotator_id, rating in enumerate(helpfulness):
                rows.append({
                    "prompt_id": pid,
                    "annotator_id": annotator_id,
                    "rating": rating,
                    "prompt": prompt,
                    "response": response
                })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved expanded ratings → {output_csv}")

if __name__ == "__main__":
    preprocess_helpsteer2("../../data/helpsteer2_disagreements.jsonl", "../../data/helpsteer2_ratings.csv")