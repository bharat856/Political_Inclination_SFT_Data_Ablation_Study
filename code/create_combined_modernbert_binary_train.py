from pathlib import Path
import pandas as pd

OLD_IN = Path("data/processed/tulu_olmo2_sft/target_domain_binary_manual_train.csv")
OOD_IN = Path("data/processed/tulu_olmo2_sft/target_domain_binary_ood_corrected.csv")

OUT = Path("data/processed/tulu_olmo2_sft/target_domain_binary_combined_train.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

old = pd.read_csv(OLD_IN)
ood = pd.read_csv(OOD_IN)

# Keep only required columns
old = old.copy()
ood = ood.copy()

if "text" not in old.columns:
    raise ValueError(f"Old file missing text column. Columns: {old.columns.tolist()}")

if "label" not in old.columns:
    raise ValueError(f"Old file missing label column. Columns: {old.columns.tolist()}")

if "text" not in ood.columns:
    raise ValueError(f"OOD file missing text column. Columns: {ood.columns.tolist()}")

if "label" not in ood.columns:
    raise ValueError(f"OOD file missing label column. Columns: {ood.columns.tolist()}")

old["source_split"] = "manual_200_old"
ood["source_split"] = "ood_299_corrected"

old_train = old[["text", "label", "source_split"]].copy()
ood_train = ood[["text", "label", "source_split"]].copy()

combined = pd.concat([old_train, ood_train], ignore_index=True)

combined["text"] = combined["text"].astype(str)
combined["label"] = combined["label"].astype(int)

combined = combined.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

combined.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(combined))
print("\nLabel counts:")
print(combined["label"].value_counts())

print("\nSource counts:")
print(combined["source_split"].value_counts())
