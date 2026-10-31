from pathlib import Path
import pandas as pd

IN = Path("outputs/modernbert_ood_test/modernbert_ood_predictions.csv")
OUT_DIR = Path("outputs/modernbert_ood_test/error_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# Normalize labels if needed
label_map = {
    0: "non_political",
    1: "political",
}

df["gold_binary_label"] = df["gold_binary"].map(label_map)
df["pred_binary_label"] = df["modernbert_pred_binary"].map(label_map)

df["error_type"] = "correct"
df.loc[(df["gold_binary"] == 0) & (df["modernbert_pred_binary"] == 1), "error_type"] = "false_positive_nonpolitical_pred_political"
df.loc[(df["gold_binary"] == 1) & (df["modernbert_pred_binary"] == 0), "error_type"] = "false_negative_political_pred_nonpolitical"

print("Error type counts:")
print(df["error_type"].value_counts())

print("\nGold manual label distribution by error type:")
if "manual_label" in df.columns:
    print(pd.crosstab(df["error_type"], df["manual_label"]))

print("\nMean political probability by error type:")
print(df.groupby("error_type")["prob_political"].describe())

# Save full error files
for error_type, g in df.groupby("error_type"):
    out = OUT_DIR / f"{error_type}.csv"
    g.to_csv(out, index=False)
    print("Saved:", out, "rows:", len(g))

# Save most important samples
fn = df[df["error_type"] == "false_negative_political_pred_nonpolitical"].copy()
fp = df[df["error_type"] == "false_positive_nonpolitical_pred_political"].copy()

fn.sort_values("prob_political").head(30).to_csv(
    OUT_DIR / "false_negative_most_confident_30.csv",
    index=False,
)

fn.sort_values("prob_political", ascending=False).head(30).to_csv(
    OUT_DIR / "false_negative_borderline_30.csv",
    index=False,
)

fp.sort_values("prob_political", ascending=False).head(30).to_csv(
    OUT_DIR / "false_positive_most_confident_30.csv",
    index=False,
)

print("\nSaved focused samples in:", OUT_DIR)
