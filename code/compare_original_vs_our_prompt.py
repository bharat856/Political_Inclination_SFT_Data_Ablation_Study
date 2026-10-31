from pathlib import Path
import argparse
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


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


def evaluate(df, gold_col, pred_col, name):
    eval_df = df[df[pred_col].notna()].copy()
    y_true = eval_df[gold_col].astype(int)
    y_pred = eval_df[pred_col].astype(int)

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)
    print("Parsed:", len(eval_df), "/", len(df))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion matrix labels [0,1,2,3]:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=["non_political", "left", "neutral", "right"],
        zero_division=0,
    ))

    print("\nPrediction distribution:")
    print(eval_df[pred_col].value_counts(dropna=False).sort_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")
    parser.add_argument("--output", default="outputs/prompt_comparison/original_vs_our_prompt_results.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print("Input:", input_path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    text_col = args.text_column or find_col(df, ["full_text", "text", "text_preview", "document", "user_text"])
    label_col = args.label_column or find_col(df, ["manual_label", "gold_label", "label", "manual_binary_label", "sft_political_label"])

    if text_col is None:
        raise ValueError("Could not find text column. Pass --text-column manually.")
    if label_col is None:
        raise ValueError("Could not find label column. Pass --label-column manually.")

    print("Using text column:", text_col)
    print("Using label column:", label_col)

    df["gold_label"] = df[label_col].apply(normalize_gold)
    df = df[df["gold_label"].notna()].copy()
    df["gold_label"] = df["gold_label"].astype(int)

    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=42).reset_index(drop=True)

    print("\nGold label distribution:")
    print(df["gold_label"].value_counts().sort_index())

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_log_stats=True,
        compilation_config=CompilationConfig(custom_ops=[]),
    )

    df = run_prompt(
        llm=llm,
        df=df,
        text_col=text_col,
        prompt_template=ORIGINAL_PROMPT,
        prompt_name="original_3way",
        allowed_labels={1, 2, 3},
        max_chars=args.max_chars,
    )

    df = run_prompt(
        llm=llm,
        df=df,
        text_col=text_col,
        prompt_template=OUR_PROMPT,
        prompt_name="our_4way",
        allowed_labels={0, 1, 2, 3},
        max_chars=args.max_chars,
    )

    evaluate(df, "gold_label", "original_3way_pred", "Original prompt: 1/2/3 only")
    evaluate(df, "gold_label", "our_4way_pred", "Our prompt: 0/1/2/3")

    df.to_csv(output_path, index=False)
    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()
