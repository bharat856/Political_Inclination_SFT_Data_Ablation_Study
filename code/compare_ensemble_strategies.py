from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE = Path("outputs/validation_tanise")

QWEN = BASE / "ensemble_v4/qwen_predictions.csv"
GEMMA = BASE / "ensemble_v4/gemma_predictions.csv"
LLAMA = BASE / (
    "slant_testset_llama_3way_dmitry_prompt/"
    "slant_testset_llama_3way_dmitry_prompt_evaluated_from_raw.csv"
)

OUT = BASE / "ensemble_v4/strategy_comparison"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["left", "neutral", "right"]


def norm(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    mapping = {
        "1": "left",
        "2": "neutral",
        "3": "right",
        "left": "left",
        "neutral": "neutral",
        "center": "neutral",
        "centre": "neutral",
        "right": "right",
    }

    return mapping.get(x)


# ------------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------------

q = pd.read_csv(QWEN)
g = pd.read_csv(GEMMA)
l = pd.read_csv(LLAMA)

if not (len(q) == len(g) == len(l) == 900):
    raise ValueError(
        f"Expected 900 rows each: "
        f"qwen={len(q)}, gemma={len(g)}, llama={len(l)}"
    )

df = pd.DataFrame({
    "text": q["text"],
    "gold": q["gold_label"].map(norm),
    "qwen": q["parsed_prediction_label"].map(norm),
    "gemma": g["parsed_prediction_label"].map(norm),
    "llama": l["pred_3way_dmitry"].map(norm),
})

if df["gold"].isna().any():
    raise ValueError("Could not parse some gold labels.")

print("Rows:", len(df))
print("Missing Qwen:", df["qwen"].isna().sum())
print("Missing Gemma:", df["gemma"].isna().sum())
print("Missing Llama:", df["llama"].isna().sum())


# ------------------------------------------------------------------
# Strategy 1: original three-model majority
# ------------------------------------------------------------------

def majority(row):
    preds = [
        p for p in [row.qwen, row.gemma, row.llama]
        if p in LABELS
    ]

    counts = {x: preds.count(x) for x in LABELS}

    winner = max(counts, key=counts.get)

    if counts[winner] >= 2:
        return winner

    # all different / insufficient consensus -> Qwen
    if row.qwen in LABELS:
        return row.qwen

    if row.gemma in LABELS:
        return row.gemma

    return row.llama


# ------------------------------------------------------------------
# Strategy 2: Tanise's Qwen + Gemma rule
# ------------------------------------------------------------------

def tanise(row):
    q = row.qwen
    g = row.gemma

    if q is None:
        return g

    if g is None:
        return q

    # Agreement
    if q == g:
        return q

    # Slant beats neutral
    if q == "neutral" and g in ("left", "right"):
        return g

    if g == "neutral" and q in ("left", "right"):
        return q

    # Left-vs-right disagreement -> Gemma tie breaker
    if {q, g} == {"left", "right"}:
        return g

    return g


# ------------------------------------------------------------------
# Strategy 3: hybrid
# ------------------------------------------------------------------

def hybrid(row):
    q = row.qwen
    g = row.gemma
    l = row.llama

    # Qwen/Gemma agreement is strong evidence
    if q == g and q in LABELS:
        return q

    if q is None:
        return g if g is not None else l

    if g is None:
        return q

    # Slant beats neutral, following Tanise
    if q == "neutral" and g in ("left", "right"):
        return g

    if g == "neutral" and q in ("left", "right"):
        return q

    # Qwen vs Gemma disagree about direction.
    # Let Llama resolve it if it agrees with one.
    if {q, g} == {"left", "right"}:
        if l == q:
            return q
        if l == g:
            return g

        # Otherwise use stronger designated tie-breaker
        return g

    return g


df["majority"] = df.apply(majority, axis=1)
df["tanise"] = df.apply(tanise, axis=1)
df["hybrid"] = df.apply(hybrid, axis=1)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

rows = []

for name in ["qwen", "gemma", "majority", "tanise", "hybrid"]:

    valid = df[name].isin(LABELS)

    gold = df.loc[valid, "gold"]
    pred = df.loc[valid, name]

    report = classification_report(
        gold,
        pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "strategy": name,
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(
            gold, pred,
            labels=LABELS,
            average="macro",
        ),
        "left_precision": report["left"]["precision"],
        "left_recall": report["left"]["recall"],
        "left_f1": report["left"]["f1-score"],
        "neutral_precision": report["neutral"]["precision"],
        "neutral_recall": report["neutral"]["recall"],
        "neutral_f1": report["neutral"]["f1-score"],
        "right_precision": report["right"]["precision"],
        "right_recall": report["right"]["recall"],
        "right_f1": report["right"]["f1-score"],
        "missing": (~valid).sum(),
    }

    rows.append(result)

    cm = confusion_matrix(
        gold,
        pred,
        labels=LABELS,
    )

    pd.DataFrame(
        cm,
        index=[f"gold_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    ).to_csv(OUT / f"confusion_matrix_{name}.csv")

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    print(
        classification_report(
            gold,
            pred,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )


summary = pd.DataFrame(rows)
summary = summary.sort_values(
    ["macro_f1", "accuracy"],
    ascending=False,
)

summary.to_csv(
    OUT / "strategy_comparison.csv",
    index=False,
)

df.to_csv(
    OUT / "all_strategy_predictions.csv",
    index=False,
)


# ------------------------------------------------------------------
# How often strategies disagree
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

cols = [
    "strategy",
    "accuracy",
    "macro_f1",
    "left_precision",
    "left_recall",
    "neutral_precision",
    "neutral_recall",
    "right_precision",
    "right_recall",
]

print(summary[cols].to_string(index=False))

print("\nStrategy disagreements:")
print(
    "Majority vs Tanise:",
    (df["majority"] != df["tanise"]).sum()
)
print(
    "Majority vs Hybrid:",
    (df["majority"] != df["hybrid"]).sum()
)
print(
    "Tanise vs Hybrid:",
    (df["tanise"] != df["hybrid"]).sum()
)

print("\nSaved to:", OUT)
