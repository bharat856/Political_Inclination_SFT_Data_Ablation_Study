from pathlib import Path
import csv
import json
import random
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]

QWEN = ROOT / (
    "outputs/dolci_olmo3_sft_slant_ensemble_mpnet_t04/"
    "qwen_predictions.jsonl"
)

GEMMA = ROOT / (
    "outputs/dolci_olmo3_sft_slant_ensemble_mpnet_t04/"
    "gemma_predictions.jsonl"
)

OUT_BASE = ROOT / (
    "data/interventions/"
    "dolci_olmo3_sft_union_mpnet_t04"
)

ROW_DIR = OUT_BASE / "row_lists"

FINAL_JSONL = OUT_BASE / "final_tanise_ensemble_labels.jsonl"
FINAL_CSV = OUT_BASE / "final_tanise_ensemble_labels.csv"
SUMMARY_JSON = OUT_BASE / "final_tanise_ensemble_summary.json"

EXPECTED_CANDIDATES = 693_639
FULL_DATASET_ROWS = 4_420_290
SEED = 42

VALID = {"left", "neutral", "right"}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def normalize(value):
    if value is None:
        return None

    value = str(value).strip().lower()

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

    return mapping.get(value)


def tanise_rule(qwen, gemma):
    # Both parsed and agree.
    if qwen == gemma and qwen in VALID:
        return qwen, "agreement"

    # One model failed: use the available valid prediction.
    if qwen not in VALID and gemma in VALID:
        return gemma, "qwen_missing_use_gemma"

    if gemma not in VALID and qwen in VALID:
        return qwen, "gemma_missing_use_qwen"

    if qwen not in VALID and gemma not in VALID:
        return None, "both_missing"

    # Slant wins over neutral.
    if qwen == "neutral" and gemma in {"left", "right"}:
        return gemma, "slant_over_neutral"

    if gemma == "neutral" and qwen in {"left", "right"}:
        return qwen, "slant_over_neutral"

    # Direct left-vs-right disagreement: Gemma tie-breaker.
    if {qwen, gemma} == {"left", "right"}:
        return gemma, "gemma_tiebreak"

    raise RuntimeError(
        f"Unexpected combination qwen={qwen!r}, gemma={gemma!r}"
    )


def write_row_list(path, rows):
    rows = sorted(rows, key=lambda x: (x["split"], x["row_id"]))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "row_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    ROW_DIR.mkdir(parents=True, exist_ok=True)

    q_iter = read_jsonl(QWEN)
    g_iter = read_jsonl(GEMMA)

    counts = Counter()
    decision_counts = Counter()
    split_label_counts = defaultdict(Counter)

    all_candidate_rows = []
    left_rows = []
    right_rows = []

    with FINAL_JSONL.open("w", encoding="utf-8") as jout, \
         FINAL_CSV.open("w", encoding="utf-8", newline="") as cout:

        fieldnames = [
            "document_id",
            "split",
            "row_id",
            "qwen_label",
            "gemma_label",
            "final_label",
            "decision_type",
        ]

        writer = csv.DictWriter(cout, fieldnames=fieldnames)
        writer.writeheader()

        n = 0

        while True:
            try:
                q = next(q_iter)
            except StopIteration:
                q = None

            try:
                g = next(g_iter)
            except StopIteration:
                g = None

            if q is None and g is None:
                break

            if q is None or g is None:
                raise RuntimeError(
                    "Qwen and Gemma files have different row counts."
                )

            q_id = str(q["document_id"])
            g_id = str(g["document_id"])

            if q_id != g_id:
                raise RuntimeError(
                    f"Alignment failure at row {n}: "
                    f"Qwen={q_id!r}, Gemma={g_id!r}"
                )

            if str(q["split"]) != str(g["split"]):
                raise RuntimeError(
                    f"Split mismatch for {q_id}"
                )

            if int(q["row_id"]) != int(g["row_id"]):
                raise RuntimeError(
                    f"row_id mismatch for {q_id}"
                )

            split = str(q["split"])
            row_id = int(q["row_id"])

            q_label = normalize(q.get("prediction_label"))
            g_label = normalize(g.get("prediction_label"))

            final_label, decision_type = tanise_rule(
                q_label,
                g_label,
            )

            record = {
                "document_id": q_id,
                "split": split,
                "row_id": row_id,
                "qwen_label": q_label,
                "gemma_label": g_label,
                "final_label": final_label,
                "decision_type": decision_type,
            }

            jout.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            writer.writerow(record)

            n += 1

            counts[final_label or "unresolved"] += 1
            decision_counts[decision_type] += 1
            split_label_counts[split][
                final_label or "unresolved"
            ] += 1

            row = {
                "split": split,
                "row_id": row_id,
            }

            # NoPol removes every political candidate identified
            # by the ModernBERT OR MPNet binary retrieval stage.
            all_candidate_rows.append(row)

            if final_label == "left":
                left_rows.append(row)

            elif final_label == "right":
                right_rows.append(row)

    if n != EXPECTED_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_CANDIDATES:,} candidates, got {n:,}"
        )

    unresolved = counts["unresolved"]

    if unresolved:
        print(
            f"WARNING: {unresolved:,} candidates have no final slant label."
        )

    # --------------------------------------------------------------
    # Intervention row lists
    # --------------------------------------------------------------

    # NoPol: remove all binary-political candidates.
    no_pol = all_candidate_rows

    # NoIdeology: remove only ideologically slanted examples.
    no_ideology = left_rows + right_rows

    # NoLeft / NoRight.
    no_left = left_rows
    no_right = right_rows

    # LRBalanced:
    # Downsample whichever ideological side is larger until
    # left/right counts are equal.
    rng = random.Random(SEED)

    if len(left_rows) > len(right_rows):
        difference = len(left_rows) - len(right_rows)
        lr_balanced = rng.sample(left_rows, difference)
        balance_removed_from = "left"

    elif len(right_rows) > len(left_rows):
        difference = len(right_rows) - len(left_rows)
        lr_balanced = rng.sample(right_rows, difference)
        balance_removed_from = "right"

    else:
        difference = 0
        lr_balanced = []
        balance_removed_from = "none"

    lists = {
        "no_pol": no_pol,
        "no_ideology": no_ideology,
        "no_left": no_left,
        "no_right": no_right,
        "lr_balanced": lr_balanced,
    }

    for name, rows in lists.items():
        path = ROW_DIR / f"remove_{name}_row_ids.csv"
        write_row_list(path, rows)
        print(f"{name:15s}: remove {len(rows):,} rows -> {path}")

    expected_rows = {
        "full": FULL_DATASET_ROWS,
        "no_pol": FULL_DATASET_ROWS - len(no_pol),
        "no_ideology": FULL_DATASET_ROWS - len(no_ideology),
        "no_left": FULL_DATASET_ROWS - len(no_left),
        "no_right": FULL_DATASET_ROWS - len(no_right),
        "lr_balanced": FULL_DATASET_ROWS - len(lr_balanced),
    }

    summary = {
        "candidate_rows": n,
        "ensemble_rule": (
            "Qwen+Gemma: agreement; slant over neutral; "
            "Gemma tie-breaks left-vs-right"
        ),
        "final_label_counts": dict(counts),
        "decision_counts": dict(decision_counts),
        "split_label_counts": {
            split: dict(c)
            for split, c in split_label_counts.items()
        },
        "left_count": len(left_rows),
        "right_count": len(right_rows),
        "lr_balance_difference": difference,
        "lr_balance_removed_from": balance_removed_from,
        "seed": SEED,
        "removal_counts": {
            name: len(rows)
            for name, rows in lists.items()
        },
        "expected_training_rows": expected_rows,
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FINAL LABEL COUNTS")
    print("=" * 100)
    for label in ["left", "neutral", "right", "unresolved"]:
        print(f"{label:12s}: {counts[label]:,}")

    print()
    print("=" * 100)
    print("EXPECTED TRAINING DATASET ROWS")
    print("=" * 100)
    for condition, rows in expected_rows.items():
        print(f"{condition:15s}: {rows:,}")

    print()
    print("Saved final labels:", FINAL_JSONL)
    print("Saved summary:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
