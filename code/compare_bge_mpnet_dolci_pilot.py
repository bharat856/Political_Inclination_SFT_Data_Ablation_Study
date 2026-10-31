from pathlib import Path
import json
import pandas as pd


BGE_PATH = Path(
    "outputs/dolci_olmo3_sft_embedding_binary_pilot_100k/"
    "dolci_first100k_embedding_modernbert_gate_comparison.csv"
)

MPNET_PATH = Path(
    "outputs/dolci_olmo3_sft_mpnet_binary_pilot_100k/"
    "dolci_first100k_embedding_modernbert_gate_comparison.csv"
)

OUT_DIR = Path("outputs/dolci_bge_mpnet_pilot_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    bge = pd.read_csv(BGE_PATH)
    mpnet = pd.read_csv(MPNET_PATH)

    keys = ["row_id"]

    for name, df in [("BGE", bge), ("MPNet", mpnet)]:
        if df["row_id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate row_id values")

    keep_bge = [
        "row_id",
        "dataset_short_name",
        "text",
        "old_modernbert_candidate",
        "embedding_prob_political",
    ]

    keep_mpnet = [
        "row_id",
        "embedding_prob_political",
    ]

    merged = bge[keep_bge].merge(
        mpnet[keep_mpnet],
        on="row_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_bge", "_mpnet"),
    )

    if len(merged) != len(bge) or len(merged) != len(mpnet):
        raise ValueError(
            f"Row mismatch: BGE={len(bge):,}, "
            f"MPNet={len(mpnet):,}, merged={len(merged):,}"
        )

    summary = {
        "rows": int(len(merged)),
        "thresholds": {},
    }

    for threshold in [0.2, 0.3, 0.4, 0.5]:
        tag = str(threshold).replace(".", "")

        bge_col = f"bge_t{tag}"
        mpnet_col = f"mpnet_t{tag}"

        merged[bge_col] = (
            merged["embedding_prob_political_bge"] >= threshold
        ).astype(int)

        merged[mpnet_col] = (
            merged["embedding_prob_political_mpnet"] >= threshold
        ).astype(int)

        merged[f"both_t{tag}"] = (
            (merged[bge_col] == 1) &
            (merged[mpnet_col] == 1)
        ).astype(int)

        merged[f"mpnet_only_t{tag}"] = (
            (merged[bge_col] == 0) &
            (merged[mpnet_col] == 1)
        ).astype(int)

        merged[f"bge_only_t{tag}"] = (
            (merged[bge_col] == 1) &
            (merged[mpnet_col] == 0)
        ).astype(int)

        merged[f"neither_t{tag}"] = (
            (merged[bge_col] == 0) &
            (merged[mpnet_col] == 0)
        ).astype(int)

        merged[f"modernbert_or_bge_t{tag}"] = (
            (merged["old_modernbert_candidate"] == 1) |
            (merged[bge_col] == 1)
        ).astype(int)

        merged[f"modernbert_or_mpnet_t{tag}"] = (
            (merged["old_modernbert_candidate"] == 1) |
            (merged[mpnet_col] == 1)
        ).astype(int)

        summary["thresholds"][str(threshold)] = {
            "bge_candidates": int(merged[bge_col].sum()),
            "mpnet_candidates": int(merged[mpnet_col].sum()),
            "mpnet_minus_bge": int(
                merged[mpnet_col].sum() - merged[bge_col].sum()
            ),
            "both": int(merged[f"both_t{tag}"].sum()),
            "mpnet_only": int(merged[f"mpnet_only_t{tag}"].sum()),
            "bge_only": int(merged[f"bge_only_t{tag}"].sum()),
            "neither": int(merged[f"neither_t{tag}"].sum()),
            "modernbert_or_bge": int(
                merged[f"modernbert_or_bge_t{tag}"].sum()
            ),
            "modernbert_or_mpnet": int(
                merged[f"modernbert_or_mpnet_t{tag}"].sum()
            ),
            "union_difference": int(
                merged[f"modernbert_or_mpnet_t{tag}"].sum()
                - merged[f"modernbert_or_bge_t{tag}"].sum()
            ),
        }

    merged_path = OUT_DIR / "bge_mpnet_first100k_merged.csv"
    summary_path = OUT_DIR / "bge_mpnet_first100k_summary.json"

    merged.to_csv(merged_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("Saved:", merged_path)
    print("Saved:", summary_path)

    # Produce manual disagreement samples at the likely final threshold.
    threshold = 0.4
    tag = "04"

    buckets = {
        "mpnet_only_t04": merged[merged[f"mpnet_only_t{tag}"] == 1],
        "bge_only_t04": merged[merged[f"bge_only_t{tag}"] == 1],
        "both_t04": merged[merged[f"both_t{tag}"] == 1],
    }

    for name, df in buckets.items():
        n = min(100, len(df))
        if n == 0:
            continue

        sample = df.sample(n=n, random_state=42).copy()
        sample["manual_binary_label"] = ""
        sample["manual_notes"] = ""

        cols = [
            "row_id",
            "dataset_short_name",
            "old_modernbert_candidate",
            "embedding_prob_political_bge",
            "embedding_prob_political_mpnet",
            "text",
            "manual_binary_label",
            "manual_notes",
        ]

        out_path = OUT_DIR / f"manual_{name}_n{n}.csv"
        sample[cols].to_csv(out_path, index=False)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
