from pathlib import Path
import pandas as pd

PRED_IN = Path("outputs/modernbert_ood_test/modernbert_ood_predictions.csv")
CORR_IN = Path("outputs/modernbert_ood_test/error_analysis/modernbert_ood_error_corrections.tsv")

OUT = Path("outputs/modernbert_ood_test/modernbert_ood_predictions_corrected.csv")
OUT_TRAIN = Path("data/processed/tulu_olmo2_sft/target_domain_binary_ood_corrected.csv")

df = pd.read_csv(PRED_IN)


def norm_label(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.lower() in {"nan", "none", ""}:
        return ""
    return s


df["manual_label_norm"] = df["manual_label"].apply(norm_label)
df["final_review_label"] = df["manual_label_norm"]
df["final_review_notes"] = ""

if not CORR_IN.exists():
    raise FileNotFoundError(f"Correction TSV not found: {CORR_IN}")

corr = pd.read_csv(CORR_IN, sep="\t", dtype=str)

required = {"row_id", "final_label", "short_note"}
missing = required - set(corr.columns)
if missing:
    raise ValueError(f"Correction TSV is missing columns: {missing}")

corr = corr.dropna(subset=["row_id", "final_label"])
corr["row_id"] = corr["row_id"].astype(str).str.strip()
corr["final_label"] = corr["final_label"].astype(str).str.strip()
corr["short_note"] = corr["short_note"].fillna("").astype(str)

valid = {"0", "1", "2", "3", "U"}
bad = corr[~corr["final_label"].isin(valid)]
if len(bad):
    print("Invalid correction labels:")
    print(bad)
    raise ValueError("Found invalid labels in correction TSV.")

df["row_id_str"] = df["row_id"].astype(str).str.strip()

corr_map = corr.drop_duplicates("row_id", keep="last").set_index("row_id")

mask = df["row_id_str"].isin(corr_map.index)

df.loc[mask, "final_review_label"] = df.loc[mask, "row_id_str"].map(
    corr_map["final_label"]
)
df.loc[mask, "final_review_notes"] = df.loc[mask, "row_id_str"].map(
    corr_map["short_note"]
)


def to_binary(label):
    label = str(label).strip()
    if label == "0":
        return 0
    if label in {"1", "2", "3"}:
        return 1
    return None


df["final_binary_label"] = df["final_review_label"].apply(to_binary)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.drop(columns=["row_id_str"]).to_csv(OUT, index=False)

train = df[df["final_binary_label"].notna()].copy()
train["label"] = train["final_binary_label"].astype(int)

train_out = train[
    ["row_id", "text_preview", "label", "final_review_label", "final_review_notes"]
].copy()

train_out = train_out.rename(columns={"text_preview": "text"})

OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
train_out.to_csv(OUT_TRAIN, index=False)

print("Corrections loaded:", len(corr))
print("Rows corrected:", int(mask.sum()))

print("Saved corrected predictions:", OUT)
print("Saved binary train file:", OUT_TRAIN)

print("\nFinal 4-way label counts:")
print(df["final_review_label"].value_counts(dropna=False))

print("\nFinal binary label counts:")
print(df["final_binary_label"].value_counts(dropna=False))
