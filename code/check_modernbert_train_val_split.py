from pathlib import Path
import pandas as pd
import re

TRAIN_FILE = Path("data/processed/tulu_olmo2_sft/target_domain_binary_manual_train.csv")
SCRIPT_FILE = Path("code/train_encoder_target_domain_binary.py")

print("=" * 80)
print("ModernBERT binary classifier train/validation split check")
print("=" * 80)

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Could not find: {TRAIN_FILE}")

df = pd.read_csv(TRAIN_FILE)

print("\nManual labelled file:")
print(TRAIN_FILE)
print("\nColumns:")
print(df.columns.tolist())

print("\nTotal labelled examples:")
print(len(df))

# Try to detect label column
possible_label_cols = ["label", "labels", "binary_label", "manual_binary_label", "label_id", "target"]
label_col = None
for c in possible_label_cols:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    # fallback: find a column with only 0/1 values
    for c in df.columns:
        vals = set(df[c].dropna().unique().tolist())
        if vals.issubset({0, 1, "0", "1"}) and len(vals) <= 2:
            label_col = c
            break

if label_col is None:
    raise ValueError("Could not automatically find label column. Check columns above.")

print(f"\nDetected label column: {label_col}")

print("\nOverall label distribution:")
print(df[label_col].value_counts(dropna=False).sort_index())

# Try to infer validation split from script
test_size = None
random_state = 42

if SCRIPT_FILE.exists():
    text = SCRIPT_FILE.read_text(errors="ignore")

    m = re.search(r"test_size\s*=\s*([0-9.]+)", text)
    if m:
        test_size = float(m.group(1))

    m = re.search(r"random_state\s*=\s*([0-9]+)", text)
    if m:
        random_state = int(m.group(1))

print("\nFrom training script:")
print("test_size =", test_size)
print("random_state =", random_state)

if test_size is None:
    print("\nCould not detect test_size from script automatically.")
    print("Open the script and check the train_test_split/test_size manually:")
    print(f"sed -n '1,220p' {SCRIPT_FILE}")
else:
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    print("\nReconstructed split:")
    print("Train examples:", len(train_df))
    print("Validation examples:", len(val_df))

    print("\nTrain label distribution:")
    print(train_df[label_col].value_counts(dropna=False).sort_index())

    print("\nValidation label distribution:")
    print(val_df[label_col].value_counts(dropna=False).sort_index())

    print("\nCopy-paste summary:")
    print(
        f"I manually labelled {len(df)} Tülu/OLMo SFT examples after excluding unclear cases. "
        f"I trained ModernBERT with a train/validation split of {len(train_df)}/{len(val_df)} examples. "
        f"The full labelled sample has label counts: {df[label_col].value_counts().sort_index().to_dict()}. "
        f"The train split has: {train_df[label_col].value_counts().sort_index().to_dict()}, "
        f"and the validation split has: {val_df[label_col].value_counts().sort_index().to_dict()}."
    )
