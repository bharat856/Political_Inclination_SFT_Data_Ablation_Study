from pathlib import Path
import json
import re

import pandas as pd


FILES = {
    "instruct": Path(
        "outputs/dolci_olmo3_sft_union_embedding_t04/4way_outputs/full_instruct/"
        "sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_"
        "hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit.csv"
    ),
    "think": Path(
        "outputs/dolci_olmo3_sft_union_embedding_t04/4way_outputs/full_think/"
        "sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_"
        "hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit.csv"
    ),
}

LABEL_MAP = {
    0: "non_political",
    1: "left",
    2: "center_neutral",
    3: "right",
}


def parse_extra(value):
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    patterns = [
        # Starts directly with a valid class.
        r"^\s*([0-3])(?:\s*[\.\):,\-]|\s|$)",
        # XML-like answer.
        r"<answer>\s*([0-3])\s*</answer>",
        # Markdown/code formatting.
        r"^\s*`{1,3}\s*([0-3])\s*`{0,3}",
        r"^\s*([0-3])\s*`{1,3}",
        # Think tags.
        r"^\s*([0-3])\s*</think>",
        r"</think>\s*([0-3])(?:\D|$)",
        # Natural-language final answer.
        r"(?:the\s+)?final\s+answer\s+is\s*:?\s*(?:\\boxed\{?|[$\\(]*)\s*([0-3])",
        r"(?:the\s+)?answer\s+is\s*:?\s*(?:\\boxed\{?|[$\\(]*)\s*([0-3])",
        r"(?:the\s+)?classification\s+is\s*:?\s*([0-3])",
        r"(?:the\s+)?label\s+is\s*:?\s*([0-3])",
        r"\bclass\s*:?\s*([0-3])\b",
        # Boxed LaTeX.
        r"\\boxed\{\s*([0-3])\s*\}",
        r"\\boxed\s*([0-3])",
    ]

    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


def reparse_file(name, path):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print("Reading:", path)

    df = pd.read_csv(path)

    old_ids = pd.to_numeric(df["sft_political_label_id"], errors="coerce")
    old_parsed = int(old_ids.notna().sum())

    reparsed_ids = []
    recovery_reason = []

    for old_id, raw in zip(old_ids, df["sft_political_raw_output"]):
        if pd.notna(old_id):
            reparsed_ids.append(int(old_id))
            recovery_reason.append("original_parser")
        else:
            recovered = parse_extra(raw)
            reparsed_ids.append(recovered)
            recovery_reason.append(
                "extended_parser" if recovered is not None else "unparsed"
            )

    df["sft_political_label_id_reparsed"] = reparsed_ids
    df["sft_political_label_reparsed"] = (
        df["sft_political_label_id_reparsed"].map(LABEL_MAP)
    )
    df["parse_ok_reparsed"] = df["sft_political_label_id_reparsed"].notna()
    df["parse_source_reparsed"] = recovery_reason

    new_parsed = int(df["parse_ok_reparsed"].sum())
    recovered = new_parsed - old_parsed

    out_csv = path.with_name(path.stem + "__reparsed.csv")
    out_summary = path.with_name(path.stem + "__reparsed_summary.json")
    out_remaining = path.with_name(path.stem + "__still_unparsed.csv")

    df.to_csv(out_csv, index=False)

    still_unparsed = df[~df["parse_ok_reparsed"]].copy()
    still_unparsed.to_csv(out_remaining, index=False)

    distribution = (
        df["sft_political_label_reparsed"]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary = {
        "split": name,
        "input": str(path),
        "rows": int(len(df)),
        "old_parsed": old_parsed,
        "new_parsed": new_parsed,
        "recovered": recovered,
        "old_parse_rate": old_parsed / len(df),
        "new_parse_rate": new_parsed / len(df),
        "still_unparsed": int(len(still_unparsed)),
        "distribution_reparsed": {
            str(k): int(v) for k, v in distribution.items()
        },
        "outputs": {
            "reparsed_csv": str(out_csv),
            "still_unparsed_csv": str(out_remaining),
            "summary_json": str(out_summary),
        },
    }

    out_summary.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nTop remaining unparsed raw outputs:")
    print(
        still_unparsed["sft_political_raw_output"]
        .value_counts(dropna=False)
        .head(30)
    )

    return summary


def main():
    summaries = {}

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        summaries[name] = reparse_file(name, path)

    combined_distribution = {
        "non_political": 0,
        "center_neutral": 0,
        "left": 0,
        "right": 0,
        "unparsed": 0,
    }

    for summary in summaries.values():
        dist = summary["distribution_reparsed"]
        combined_distribution["non_political"] += dist.get("non_political", 0)
        combined_distribution["center_neutral"] += dist.get("center_neutral", 0)
        combined_distribution["left"] += dist.get("left", 0)
        combined_distribution["right"] += dist.get("right", 0)
        combined_distribution["unparsed"] += dist.get("nan", 0)

    total_rows = sum(s["rows"] for s in summaries.values())
    total_parsed = sum(s["new_parsed"] for s in summaries.values())
    total_recovered = sum(s["recovered"] for s in summaries.values())

    combined = {
        "rows": total_rows,
        "parsed": total_parsed,
        "parse_rate": total_parsed / total_rows,
        "recovered_by_extended_parser": total_recovered,
        "distribution": combined_distribution,
        "confirmed_political": (
            combined_distribution["center_neutral"]
            + combined_distribution["left"]
            + combined_distribution["right"]
        ),
        "ideological": (
            combined_distribution["left"]
            + combined_distribution["right"]
        ),
    }

    out = Path(
        "outputs/dolci_olmo3_sft_union_embedding_t04/4way_outputs/"
        "dolci_union_t04_4way_reparsed_combined_summary.json"
    )
    out.write_text(
        json.dumps(
            {
                "splits": summaries,
                "combined": combined,
            },
            indent=2,
        )
    )

    print("\n" + "=" * 100)
    print("COMBINED")
    print("=" * 100)
    print(json.dumps(combined, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
