from pathlib import Path
import json
import pandas as pd

FILES = [
    {
        "name": "dolci_instruct_sft_7b",
        "inp": Path("outputs/dolci_olmo3_sft_modernbert_combined/dolci_instruct_sft_7b_political_candidates.jsonl"),
        "out": Path("outputs/dolci_olmo3_sft_modernbert_combined/4way_input/dolci_instruct_sft_7b_political_candidates_for_4way.csv"),
    },
    {
        "name": "dolci_think_sft_7b",
        "inp": Path("outputs/dolci_olmo3_sft_modernbert_combined/dolci_think_sft_7b_political_candidates.jsonl"),
        "out": Path("outputs/dolci_olmo3_sft_modernbert_combined/4way_input/dolci_think_sft_7b_political_candidates_for_4way.csv"),
    },
]

for cfg in FILES:
    rows = []
    print("Reading:", cfg["inp"])

    with cfg["inp"].open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            rows.append({
                "row_id": obj.get("row_id"),
                "dataset": obj.get("dataset"),
                "dataset_short_name": obj.get("dataset_short_name"),
                "dataset_source": obj.get("dataset_source", ""),
                "id": obj.get("id", ""),
                "modernbert_binary_pred": obj.get("modernbert_binary_pred"),
                "modernbert_prob_political": obj.get("modernbert_prob_political"),
                "full_text": obj.get("text", ""),
            })

    df = pd.DataFrame(rows)
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["out"], index=False)

    print("Saved:", cfg["out"])
    print("Rows:", len(df))
    print(df[["modernbert_binary_pred", "modernbert_prob_political"]].describe(include="all"))
