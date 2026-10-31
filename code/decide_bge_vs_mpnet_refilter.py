from pathlib import Path
import json
import pandas as pd


PATHS = {
    "pilot": Path(
        "outputs/dolci_bge_mpnet_pilot_comparison/"
        "bge_mpnet_first100k_summary.json"
    ),
    "external_comparison": Path(
        "outputs/validation_external/model_comparisons/"
        "bge_vs_mpnet_binary_comparison.csv"
    ),
    "manual_mpnet": Path(
        "outputs/validation_target_domain/"
        "manual_binary_gate_comparison_mpnet/"
        "manual_target_domain_binary_gate_summary.json"
    ),
    "manual_bge": Path(
        "outputs/validation_target_domain/"
        "manual_binary_gate_comparison/"
        "manual_target_domain_binary_gate_summary.json"
    ),
    "majority": Path(
        "outputs/validation_baselines/"
        "majority_baselines_summary.csv"
    ),
}

OUT_DIR = Path("outputs/dolci_bge_mpnet_pilot_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def get_method(summary: dict, name: str) -> dict:
    methods = summary.get("methods", {})
    if name not in methods:
        raise KeyError(
            f"Method {name!r} not found. "
            f"Available: {list(methods)}"
        )
    return methods[name]


def main():
    pilot = load_json(PATHS["pilot"])
    external = pd.read_csv(PATHS["external_comparison"])
    manual_mpnet = load_json(PATHS["manual_mpnet"])
    majority = pd.read_csv(PATHS["majority"])

    manual_bge = None
    if PATHS["manual_bge"].exists():
        manual_bge = load_json(PATHS["manual_bge"])

    threshold = "0.4"

    if threshold not in pilot["thresholds"]:
        raise KeyError(
            f"Threshold {threshold} missing from pilot. "
            f"Available: {list(pilot['thresholds'])}"
        )

    pilot_t04 = pilot["thresholds"][threshold]
    rows = int(pilot["rows"])

    # External benchmark improvements
    ext_rows = external[
        external["metric"].isin([
            "accuracy",
            "macro_f1",
            "political_precision",
            "political_recall",
            "political_f1",
        ])
    ].copy()

    pivot = ext_rows.pivot(
        index="dataset",
        columns="metric",
        values="mpnet_minus_bge",
    )

    # Manual target-domain MPNet results
    mpnet_embedding = get_method(
        manual_mpnet,
        "embedding_t0.4",
    )
    mpnet_union = get_method(
        manual_mpnet,
        "union_t0.4",
    )
    modernbert = get_method(
        manual_mpnet,
        "modernbert",
    )

    bge_embedding = None
    bge_union = None

    if manual_bge is not None:
        if "embedding_t0.4" in manual_bge.get("methods", {}):
            bge_embedding = get_method(
                manual_bge,
                "embedding_t0.4",
            )
        if "union_t0.4" in manual_bge.get("methods", {}):
            bge_union = get_method(
                manual_bge,
                "union_t0.4",
            )

    union_difference = int(pilot_t04["union_difference"])
    union_difference_rate = union_difference / rows

    mpnet_only = int(pilot_t04["mpnet_only"])
    bge_only = int(pilot_t04["bge_only"])
    disagreement = mpnet_only + bge_only
    disagreement_rate = disagreement / rows

    mpnet_candidates = int(pilot_t04["mpnet_candidates"])
    bge_candidates = int(pilot_t04["bge_candidates"])

    mb_or_mpnet = int(
        pilot_t04["modernbert_or_mpnet"]
    )
    mb_or_bge = int(
        pilot_t04["modernbert_or_bge"]
    )

    # Transparent decision signals
    external_f1_better_all = bool(
        (pivot["political_f1"] > 0).all()
    )
    external_macro_better_all = bool(
        (pivot["macro_f1"] > 0).all()
    )

    external_recall_acceptable = bool(
        (pivot["political_recall"] >= -0.01).all()
    )

    target_union_recall_good = (
        mpnet_union["political_recall"] >= 0.95
    )
    target_union_precision_usable = (
        mpnet_union["political_precision"] >= 0.60
    )

    # Candidate-set impact thresholds:
    # >= 1% of the pilot = meaningful pipeline change
    # < 0.5% = probably too small to justify a full rerun
    meaningful_union_change = abs(
        union_difference_rate
    ) >= 0.01

    negligible_union_change = abs(
        union_difference_rate
    ) < 0.005

    clear_validation_gain = (
        external_f1_better_all
        and external_macro_better_all
        and external_recall_acceptable
    )

    target_gate_acceptable = (
        target_union_recall_good
        and target_union_precision_usable
    )

    if (
        clear_validation_gain
        and target_gate_acceptable
        and meaningful_union_change
    ):
        decision = "REFILTER_FULL_DOLCI"
        explanation = (
            "MPNet improves the external validation metrics, "
            "the MPNet union retains at least 95% political "
            "recall with usable precision on the target-domain "
            "validation set, and it materially changes the "
            "100k DOLCI candidate union."
        )
    elif negligible_union_change:
        decision = "KEEP_EXISTING_INTERVENTIONS"
        explanation = (
            "Although MPNet validates better, it changes less "
            "than 0.5% of the 100k DOLCI union. Given the thesis "
            "timeline, the practical gain is too small to justify "
            "re-filtering 4.42 million rows."
        )
    else:
        decision = "KEEP_EXISTING_AND_START_SFT"
        explanation = (
            "MPNet is quantitatively better, but the evidence is "
            "not strong enough to justify delaying training without "
            "manual disagreement validation. Given the current time "
            "constraint, retain the validated intervention datasets "
            "and start the OLMo-3 runs."
        )

    report = {
        "decision": decision,
        "decision_explanation": explanation,
        "decision_rule": {
            "meaningful_union_change": "absolute union change >= 1.0% of pilot",
            "negligible_union_change": "absolute union change < 0.5% of pilot",
            "target_union_min_recall": 0.95,
            "target_union_min_precision": 0.60,
            "maximum_acceptable_external_recall_drop": 0.01,
        },
        "pilot_t0.4": {
            "rows": rows,
            "bge_embedding_candidates": bge_candidates,
            "mpnet_embedding_candidates": mpnet_candidates,
            "mpnet_minus_bge_embedding_candidates": (
                mpnet_candidates - bge_candidates
            ),
            "modernbert_or_bge": mb_or_bge,
            "modernbert_or_mpnet": mb_or_mpnet,
            "union_difference": union_difference,
            "union_difference_rate": union_difference_rate,
            "both": int(pilot_t04["both"]),
            "mpnet_only": mpnet_only,
            "bge_only": bge_only,
            "disagreement_rows": disagreement,
            "disagreement_rate": disagreement_rate,
        },
        "manual_target_domain_t0.4": {
            "modernbert": {
                "precision": modernbert["political_precision"],
                "recall": modernbert["political_recall"],
                "f1": modernbert["political_f1"],
                "accuracy": modernbert["accuracy"],
            },
            "mpnet_embedding": {
                "precision": mpnet_embedding["political_precision"],
                "recall": mpnet_embedding["political_recall"],
                "f1": mpnet_embedding["political_f1"],
                "accuracy": mpnet_embedding["accuracy"],
            },
            "modernbert_or_mpnet": {
                "precision": mpnet_union["political_precision"],
                "recall": mpnet_union["political_recall"],
                "f1": mpnet_union["political_f1"],
                "accuracy": mpnet_union["accuracy"],
            },
        },
        "external_mpnet_minus_bge": (
            pivot.fillna(0).to_dict(orient="index")
        ),
        "signals": {
            "external_political_f1_improved_on_all": (
                external_f1_better_all
            ),
            "external_macro_f1_improved_on_all": (
                external_macro_better_all
            ),
            "external_recall_drop_never_exceeds_1pp": (
                external_recall_acceptable
            ),
            "target_union_recall_at_least_95_percent": (
                target_union_recall_good
            ),
            "target_union_precision_at_least_60_percent": (
                target_union_precision_usable
            ),
            "pilot_union_change_at_least_1_percent": (
                meaningful_union_change
            ),
            "pilot_union_change_below_half_percent": (
                negligible_union_change
            ),
        },
    }

    if bge_embedding is not None:
        report["manual_target_domain_t0.4"][
            "bge_embedding"
        ] = {
            "precision": bge_embedding["political_precision"],
            "recall": bge_embedding["political_recall"],
            "f1": bge_embedding["political_f1"],
            "accuracy": bge_embedding["accuracy"],
        }

    if bge_union is not None:
        report["manual_target_domain_t0.4"][
            "modernbert_or_bge"
        ] = {
            "precision": bge_union["political_precision"],
            "recall": bge_union["political_recall"],
            "f1": bge_union["political_f1"],
            "accuracy": bge_union["accuracy"],
        }

    out_json = OUT_DIR / "bge_mpnet_quantitative_decision.json"
    out_json.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("BGE VS MPNET: QUANTITATIVE DECISION")
    print("=" * 100)

    print("\n100K DOLCI PILOT — THRESHOLD 0.4")
    print(f"Rows:                          {rows:,}")
    print(f"BGE embedding candidates:      {bge_candidates:,}")
    print(f"MPNet embedding candidates:    {mpnet_candidates:,}")
    print(
        "Embedding candidate change:    "
        f"{mpnet_candidates - bge_candidates:+,}"
    )
    print(f"ModernBERT OR BGE:             {mb_or_bge:,}")
    print(f"ModernBERT OR MPNet:           {mb_or_mpnet:,}")
    print(
        "Union change:                  "
        f"{union_difference:+,} "
        f"({pct(union_difference_rate)})"
    )
    print(f"MPNet-only rows:               {mpnet_only:,}")
    print(f"BGE-only rows:                 {bge_only:,}")
    print(
        "Total embedding disagreement:  "
        f"{disagreement:,} "
        f"({pct(disagreement_rate)})"
    )

    print("\nMANUAL TARGET-DOMAIN VALIDATION")
    print(
        "ModernBERT:                    "
        f"P={modernbert['political_precision']:.3f}  "
        f"R={modernbert['political_recall']:.3f}  "
        f"F1={modernbert['political_f1']:.3f}"
    )
    print(
        "MPNet t0.4:                    "
        f"P={mpnet_embedding['political_precision']:.3f}  "
        f"R={mpnet_embedding['political_recall']:.3f}  "
        f"F1={mpnet_embedding['political_f1']:.3f}"
    )
    print(
        "ModernBERT OR MPNet t0.4:       "
        f"P={mpnet_union['political_precision']:.3f}  "
        f"R={mpnet_union['political_recall']:.3f}  "
        f"F1={mpnet_union['political_f1']:.3f}"
    )

    print("\nEXTERNAL VALIDATION: MPNET MINUS BGE")
    for dataset, values in pivot.iterrows():
        print(
            f"{dataset:20s} "
            f"accuracy={pp(values['accuracy'])}, "
            f"macro-F1={pp(values['macro_f1'])}, "
            f"political-F1={pp(values['political_f1'])}, "
            f"recall={pp(values['political_recall'])}"
        )

    print("\n" + "-" * 100)
    print("DECISION:", decision)
    print(explanation)
    print("-" * 100)

    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()
