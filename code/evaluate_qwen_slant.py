from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

INPUT = Path(
    "outputs/validation_tanise/ensemble_v4/qwen_predictions.csv"
)

OUT_DIR = Path(
    "outputs/validation_tanise/ensemble_v4/evaluation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["left", "neutral", "right"]

df = pd.read_csv(INPUT)

df = df[df["parsed_prediction_label"].notna()].copy()

gold = df["gold_label"].str.lower()
pred = df["parsed_prediction_label"].str.lower()

accuracy = accuracy_score(gold, pred)

report = classification_report(
    gold,
    pred,
    labels=LABELS,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    gold,
    pred,
    labels=LABELS,
)

summary = {
    "rows": len(df),
    "accuracy": accuracy,
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "left_precision": report["left"]["precision"],
    "left_recall": report["left"]["recall"],
    "left_f1": report["left"]["f1-score"],
    "neutral_precision": report["neutral"]["precision"],
    "neutral_recall": report["neutral"]["recall"],
    "neutral_f1": report["neutral"]["f1-score"],
    "right_precision": report["right"]["precision"],
    "right_recall": report["right"]["recall"],
    "right_f1": report["right"]["f1-score"],
}

print("=" * 100)
print("QWEN TANISE RESULTS")
print("=" * 100)

for k, v in summary.items():
    print(f"{k:20s}: {v}")

print()
print(classification_report(
    gold,
    pred,
    labels=LABELS,
    zero_division=0,
))

print("Confusion matrix")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS,
).to_csv(
    OUT_DIR / "confusion_matrix.csv"
)

(Path(OUT_DIR) / "classification_report.txt").write_text(
    classification_report(
        gold,
        pred,
        labels=LABELS,
        zero_division=0,
    )
)

(OUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

print()
print("Saved:", OUT_DIR)
