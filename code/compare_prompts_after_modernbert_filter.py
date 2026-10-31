from pathlib import Path
import argparse
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from vllm import LLM, SamplingParams


ORIGINAL_PROMPT = """### PROMPT:
You are an expert in political language. Classify the document below as LEFT (1), NEUTRAL (2), or RIGHT (3) based on:
- **Language** (partisan terms)
- **Position** (alignment with progressive or conservative policies)
- **Framing** (balance vs. one-sidedness)
Use only the content of the document. Ignore source or author.
Respond with one number: 1 (LEFT), 2 (NEUTRAL), or 3 (RIGHT).
### DOCUMENT:
{document}
### ANSWER:"""


OUR_PROMPT = """### PROMPT:
You are an expert in political language. Classify the document below into exactly one of four labels:

0 = NON-POLITICAL
The document does not contain a real political issue, political actor, public policy, government action, election, ideology, or political conflict. Also use this label for normal coding, math, translation, formatting, title-generation, extraction, or general tasks where political words are only incidental.

1 = LEFT
The document expresses or strongly frames a progressive, social-democratic, egalitarian, pro-welfare, pro-worker, anti-discrimination, climate-action, minority-rights, or similar left/progressive position.

2 = NEUTRAL
The document is political, but mainly factual, descriptive, balanced, procedural, historical, or does not clearly support a left or right position.

3 = RIGHT
The document expresses or strongly frames a conservative, nationalist, free-market, anti-regulation, law-and-order, traditionalist, stricter-immigration, or similar right/conservative position.

Use only the content of the document. Ignore source or author.
Respond with one number only: 0, 1, 2, or 3.

### DOCUMENT:
{document}

### ANSWER:"""


def parse_label(raw, allowed):
    if raw is None:
        return None
    s = str(raw).strip()
    m = re.search(r"\b([0-3])\b", s)
    if not m:
        return None
    y = int(m.group(1))
    if y not in allowed:
        return None
    return y


def normalize_gold(x):
    s = str(x).strip().lower()
    mapping = {
        "0": 0,
        "non_political": 0,
        "non-political": 0,
        "non political": 0,
        "1": 1,
        "left": 1,
        "2": 2,
        "center": 2,
        "centre": 2,
        "neutral": 2,
        "center_neutral": 2,
        "center/neutral": 2,
        "3": 3,
        "right": 3,
    }
    return mapping.get(s, None)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def run_prompt(llm, df, text_col, prompt_template, prompt_name, allowed_labels, max_chars):
    prompts = []
    for text in df[text_col].tolist():
        text = str(text)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        prompts.append(prompt_template.format(document=text))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
    )

    outputs = llm.generate(prompts, sampling_params)

    raw_outputs = []
    preds = []

    for out in outputs:
        raw = out.outputs[0].text.strip() if out.outputs else ""
        pred = parse_label(raw, allowed=allowed_labels)
        raw_outputs.append(raw)
        preds.append(pred)

    df[f"{prompt_name}_raw"] = raw_outputs
    df[f"{prompt_name}_pred"] = preds
    return df


def evaluate(df, gold_col, pred_col, name, labels):
    eval_df = df[df[pred_col].notna()].copy()
    y_true = eval_df[gold_col].astype(int)
    y_pred = eval_df[pred_col].astype(int)

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print("Parsed:", len(eval_df), "/", len(df))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion matrix labels:", labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))

    target_names = []
    for l in labels:
        if l == 0:
            target_names.append("non_political")
        elif l == 1:
            target_names.append("left")
        elif l == 2:
            target_names.append("neutral")
        elif l == 3:
            target_names.append("right")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    ))

    print("\nPrediction distribution:")
    print(eval_df[pred_col].value_counts(dropna=False).sort_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-input", default="outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")
    parser.add_argument("--modernbert-preds", default="outputs/target_domain_binary_modernbert/tulu_olmo2_sft_candidates_binary_predictions.csv")
    parser.add_argument("--output", default="outputs/prompt_comparison/two_stage_prompt_comparison_results.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text-column", default="text_preview")
    parser.add_argument("--label-column", default="manual_label")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = parser.parse_args()

    manual_path = Path(args.manual_input)
    modernbert_path = Path(args.modernbert_preds)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manual_df = pd.read_csv(manual_path)
    mb_df = pd.read_csv(modernbert_path)

    print("Manual file:", manual_path)
    print("Manual rows:", len(manual_df))
    print("ModernBERT prediction file:", modernbert_path)
    print("ModernBERT rows:", len(mb_df))

    if "row_id" not in manual_df.columns or "row_id" not in mb_df.columns:
        raise ValueError("Both files must contain row_id.")

    if "binary_pred_label" not in mb_df.columns:
        raise ValueError("ModernBERT prediction file must contain binary_pred_label.")

    keep_cols = ["row_id", "binary_pred_label"]
    if "prob_political" in mb_df.columns:
        keep_cols.append("prob_political")
    if "prob_non_political" in mb_df.columns:
        keep_cols.append("prob_non_political")

    df = manual_df.merge(mb_df[keep_cols], on="row_id", how="left")

    df["gold_label"] = df[args.label_column].apply(normalize_gold)
    df = df[df["gold_label"].notna()].copy()
    df["gold_label"] = df["gold_label"].astype(int)

    print("\nBefore ModernBERT filter:")
    print("Rows:", len(df))
    print(df["gold_label"].value_counts().sort_index())

    filtered = df[df["binary_pred_label"] == "political"].copy().reset_index(drop=True)

    print("\nAfter ModernBERT filter:")
    print("Rows:", len(filtered))
    print("Gold distribution after filter:")
    print(filtered["gold_label"].value_counts().sort_index())
    print("\nModernBERT predicted label distribution on manual sample:")
    print(df["binary_pred_label"].value_counts(dropna=False))

    if len(filtered) == 0:
        raise ValueError("No rows passed ModernBERT filter.")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    filtered = run_prompt(
        llm=llm,
        df=filtered,
        text_col=args.text_column,
        prompt_template=ORIGINAL_PROMPT,
        prompt_name="original_3way_after_mb",
        allowed_labels={1, 2, 3},
        max_chars=args.max_chars,
    )

    filtered = run_prompt(
        llm=llm,
        df=filtered,
        text_col=args.text_column,
        prompt_template=OUR_PROMPT,
        prompt_name="our_4way_after_mb",
        allowed_labels={0, 1, 2, 3},
        max_chars=args.max_chars,
    )

    # Evaluate full 4-label outcome after the two-stage pipeline:
    # Any rows not passing ModernBERT are final 0.
    final_df = df.copy()

    original_map = filtered.set_index("row_id")["original_3way_after_mb_pred"].to_dict()
    our_map = filtered.set_index("row_id")["our_4way_after_mb_pred"].to_dict()

    final_df["two_stage_original_prompt_pred"] = final_df.apply(
        lambda r: original_map.get(r["row_id"], 0), axis=1
    )

    final_df["two_stage_our_prompt_pred"] = final_df.apply(
        lambda r: our_map.get(r["row_id"], 0), axis=1
    )

    print("\n\nTWO-STAGE FINAL EVALUATION")
    evaluate(
        final_df,
        "gold_label",
        "two_stage_original_prompt_pred",
        "Two-stage: ModernBERT filter + original 3-way prompt",
        labels=[0, 1, 2, 3],
    )

    evaluate(
        final_df,
        "gold_label",
        "two_stage_our_prompt_pred",
        "Two-stage: ModernBERT filter + revised 4-way prompt",
        labels=[0, 1, 2, 3],
    )

    print("\n\nPROMPT-ONLY EVALUATION ON MODERNBERT-PASSED ROWS")
    evaluate(
        filtered,
        "gold_label",
        "original_3way_after_mb_pred",
        "Only rows passed by ModernBERT: original 3-way prompt",
        labels=[1, 2, 3],
    )

    evaluate(
        filtered,
        "gold_label",
        "our_4way_after_mb_pred",
        "Only rows passed by ModernBERT: revised 4-way prompt",
        labels=[0, 1, 2, 3],
    )

    filtered.to_csv(output_path, index=False)
    final_path = output_path.with_name(output_path.stem + "_final_predictions.csv")
    final_df.to_csv(final_path, index=False)

    print("\nSaved filtered prompt results to:", output_path)
    print("Saved final two-stage predictions to:", final_path)


if __name__ == "__main__":
    main()
