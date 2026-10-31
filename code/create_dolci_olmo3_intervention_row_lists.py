from pathlib import Path
import pandas as pd
import json

CLS = Path("outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_binary_candidates_4way_combined.csv")
OUT_DIR = Path("data/interventions/dolci_olmo3_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLS)

# Normalize
df["label"] = df["final_political_4way_status"].fillna("unparsed")

confirmed_political = {"left", "right", "center_neutral"}
ideological = {"left", "right"}

remove_no_pol = df[df["label"].isin(confirmed_political)]
remove_no_ideology = df[df["label"].isin(ideological)]
remove_left = df[df["label"] == "left"]
remove_right = df[df["label"] == "right"]

# Balanced LR: remove extra left rows, keep all right rows.
left = df[df["label"] == "left"].copy()
right = df[df["label"] == "right"].copy()

n_right = len(right)
left_keep = left.sample(n=n_right, random_state=42)
left_remove_for_balance = left.drop(index=left_keep.index)

row_lists = {
    "remove_no_pol": remove_no_pol,
    "remove_no_ideology": remove_no_ideology,
    "remove_left": remove_left,
    "remove_right": remove_right,
    "remove_extra_left_for_lr_balance": left_remove_for_balance,
}

metadata = {}

for name, sub in row_lists.items():
    out = OUT_DIR / f"{name}_row_ids.csv"
    cols = ["dolci_split", "dataset_short_name", "row_id", "id", "label"]
    cols = [c for c in cols if c in sub.columns]
    sub[cols].to_csv(out, index=False)

    metadata[name] = {
        "rows": int(len(sub)),
        "label_distribution": sub["label"].value_counts(dropna=False).to_dict(),
        "by_split": sub["dolci_split"].value_counts(dropna=False).to_dict() if "dolci_split" in sub.columns else {},
        "file": str(out),
    }

meta_out = OUT_DIR / "intervention_row_lists_metadata.json"
meta_out.write_text(json.dumps(metadata, indent=2))

print("Saved row lists to:", OUT_DIR)
print(json.dumps(metadata, indent=2))
