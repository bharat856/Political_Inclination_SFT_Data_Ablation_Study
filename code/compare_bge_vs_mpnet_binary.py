from pathlib import Path
import json
import pandas as pd


BGE_DIR = Path(
    "outputs/validation_external/embedding_binary_20ng_bbc"
)

MPNET_DIR = Path(
    "outputs/validation_external/mpnet_binary_20ng_bbc"
)

DATASETS = [
    "20newsgroups",
    "bbc_news",
]


def load_summary(base_dir: Path, dataset: str) -> dict:
    path = base_dir / f"{dataset}_embedding_binary_cv_summary.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_metric(summary: dict, candidates: list[str]):
    for key in candidates:
        if key in summary:
            return summary[key]

    for value in summary.values():
        if isinstance(value, dict):
            result = find_metric(value, candidates)
            if result is not None:
                return result

    return None


def extract_metrics(summary: dict) -> dict:
    return {
        "rows": find_metric(
            summary,
            ["rows", "n_rows", "num_rows"],
        ),
        "accuracy": find_metric(
            summary,
            ["accuracy", "cv_accuracy"],
        ),
        "macro_f1": find_metric(
            summary,
            ["macro_f1", "macro avg_f1-score"],
        ),
        "weighted_f1": find_metric(
            summary,
            ["weighted_f1", "weighted avg_f1-score"],
        ),
        "political_precision": find_metric(
            summary,
            [
                "political_precision",
                "precision_political",
            ],
        ),
        "political_recall": find_metric(
            summary,
            [
                "political_recall",
                "recall_political",
            ],
        ),
        "political_f1": find_metric(
            summary,
            [
                "political_f1",
                "f1_political",
            ],
        ),
        "confusion_matrix": find_metric(
            summary,
            ["confusion_matrix"],
        ),
    }


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    rows = []

    for dataset in DATASETS:
        bge_summary = load_summary(BGE_DIR, dataset)
        mpnet_summary = load_summary(MPNET_DIR, dataset)

        bge = extract_metrics(bge_summary)
        mpnet = extract_metrics(mpnet_summary)

        print("\n" + "=" * 100)
        print(dataset)
        print("=" * 100)

        print("\nBGE raw summary:")
        print(json.dumps(bge_summary, indent=2))

        print("\nMPNet raw summary:")
        print(json.dumps(mpnet_summary, indent=2))

        for metric in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "political_precision",
            "political_recall",
            "political_f1",
        ]:
            bge_value = as_float(bge[metric])
            mpnet_value = as_float(mpnet[metric])

            difference = None
            if bge_value is not None and mpnet_value is not None:
                difference = mpnet_value - bge_value

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "bge": bge_value,
                "mpnet": mpnet_value,
                "mpnet_minus_bge": difference,
            })

        print("\nExtracted metrics:")
        print("BGE:", json.dumps(bge, indent=2))
        print("MPNet:", json.dumps(mpnet, indent=2))

    comparison = pd.DataFrame(rows)

    out_dir = Path("outputs/validation_external/model_comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "bge_vs_mpnet_binary_comparison.csv"
    comparison.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("COMBINED COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))
    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()
