import re
from pathlib import Path
import pandas as pd

FILES = [
    "outputs/benchmark_v3a_binary/binary_classifier_v2__cyankiwi_Qwen3_5-27B-AWQ-BF16-INT8__test.csv",
    "outputs/benchmark_v3a_binary/binary_classifier__cyankiwi_Qwen3_5-27B-AWQ-BF16-INT8__test.csv",
    "outputs/benchmark_v3a_binary/binary_classifier__cyankiwi_gemma-4-31B-it-AWQ-8bit__test.csv",
    "outputs/benchmark_v3a_binary/binary_classifier_v2__cyankiwi_gemma-4-31B-it-AWQ-8bit__test.csv",
]

def parse(raw):
    s = str(raw).strip()
    low = s.lower()

    # Remove think tags but keep text inside for numeric/label search.
    low = low.replace("<think>", " ").replace("</think>", " ")

    # Strong exact starts
    stripped = low.strip()
    if stripped.startswith("left"):
        return "left"
    if stripped.startswith("right"):
        return "right"

    # If first visible generated token is numeric
    m = re.search(r"\b([12])\b", stripped)
    if m:
        return "left" if m.group(1) == "1" else "right"

    # Find labels anywhere
    label_matches = re.findall(r"\b(left|right|liberal|progressive|conservative)\b", stripped)
    if label_matches:
        first = label_matches[0]
        if first in ["left", "liberal", "progressive"]:
            return "left"
        if first in ["right", "conservative"]:
            return "right"

    return None

for f in FILES:
    p = Path(f)
    if not p.exists():
        continue

    df = pd.read_csv(p)
    df["reparsed_prediction_label"] = df["raw_output"].apply(parse)
    df["reparse_ok"] = df["reparsed_prediction_label"].notna()
    df["reparse_correct"] = df["reparsed_prediction_label"] == df["label"]

    out = p.with_name(p.stem + "__reparsed.csv")
    df.to_csv(out, index=False)

    parsed_df = df[df["reparse_ok"]]

    print("\n" + "="*100)
    print(p.name)
    print("Saved:", out)
    print("Rows:", len(df))
    print("Reparsed:", int(df["reparse_ok"].sum()), "/", len(df))
    print("Accuracy all rows:", round(df["reparse_correct"].mean(), 4))
    if len(parsed_df):
        print("Accuracy parsed only:", round(parsed_df["reparse_correct"].mean(), 4))

    print("\nPredicted:")
    print(df["reparsed_prediction_label"].value_counts(dropna=False))

    print("\nConfusion:")
    print(pd.crosstab(df["label"], df["reparsed_prediction_label"], rownames=["gold"], colnames=["pred"]))
