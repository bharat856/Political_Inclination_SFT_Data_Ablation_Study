from pathlib import Path
import pandas as pd

candidate_path = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_political_candidates.jsonl")
old_manual_path = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")
out_path = Path("outputs/modernbert_ood_test/manual_review_ood_sample_300.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

sample_n = 300
random_state = 123

candidates = pd.read_json(candidate_path, lines=True)
old = pd.read_csv(old_manual_path)

old_ids = set(old["row_id"].tolist())
new_pool = candidates[~candidates["row_id"].isin(old_ids)].copy()

# Prefer a useful review view
cols = []
for c in [
    "row_id", "id", "source", "dataset",
    "strong_hits", "issue_hits", "context_hits",
    "full_text", "user_text", "assistant_text"
]:
    if c in new_pool.columns:
        cols.append(c)

sample = new_pool.sample(n=sample_n, random_state=random_state).copy()

if "full_text" in sample.columns:
    sample["text_preview"] = sample["full_text"].astype(str).str.slice(0, 2500)
elif "user_text" in sample.columns:
    sample["text_preview"] = sample["user_text"].astype(str).str.slice(0, 2500)

sample["manual_label"] = ""
sample["manual_notes"] = ""

keep = [c for c in ["row_id", "id", "source", "dataset", "text_preview", "manual_label", "manual_notes"] if c in sample.columns]
sample[keep].to_csv(out_path, index=False)

print("Candidate rows:", len(candidates))
print("Old manual row_ids excluded:", len(old_ids))
print("New pool rows:", len(new_pool))
print("Saved:", out_path)
print("Rows:", len(sample))
