import argparse
import re
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """Classify the political ideology of the following text.

Allowed labels:
LEFT
RIGHT

Return exactly one label: LEFT or RIGHT.
Do not explain.

Text:
{text}

Label:"""

def parse_prediction(s):
    raw = str(s).strip()
    low = raw.lower().strip()

    # Remove common wrappers
    low = low.replace("label:", "").replace("answer:", "").strip()
    low = low.replace("**", "").replace(".", "").strip()

    # Exact/starting matches
    if low.startswith("left"):
        return "left"
    if low.startswith("right"):
        return "right"

    # Numeric fallback
    if low.startswith("1"):
        return "left"
    if low.startswith("2"):
        return "right"

    # Search first meaningful label
    matches = re.findall(r"\b(left|right|liberal|progressive|conservative)\b", low)
    if matches:
        first = matches[0]
        if first in ["left", "liberal", "progressive"]:
            return "left"
        if first in ["right", "conservative"]:
            return "right"

    return None

def safe_model_name(model):
    return (
        model.replace("/", "_")
             .replace(":", "_")
             .replace(".", "_")
             .replace(" ", "_")
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.benchmark)

    if "source_split" in df.columns:
        df = df[df["source_split"] == args.split].copy()

    if args.limit:
        df = df.head(args.limit).copy()

    df = df.reset_index(drop=True)

    print("Rows to run:", len(df))
    print("Model:", args.model)
    print("Gold:")
    print(df["label"].value_counts())

    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = [PROMPT_TEMPLATE.format(text=str(t)) for t in df["text"]]

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    raw_outputs = [o.outputs[0].text.strip() for o in outputs]
    preds = [parse_prediction(x) for x in raw_outputs]

    df["raw_output"] = raw_outputs
    df["parsed_prediction_label"] = preds
    df["parse_ok"] = df["parsed_prediction_label"].notna()
    df["is_correct"] = df["parsed_prediction_label"] == df["label"]

    out_path = out_dir / f"binary_classifier_v2__{safe_model_name(args.model)}__{args.split}.csv"
    df.to_csv(out_path, index=False)

    parsed_df = df[df["parse_ok"]].copy()

    print("Saved:", out_path)
    print("Parsed:", int(df["parse_ok"].sum()), "/", len(df))
    print("Accuracy all rows:", round(df["is_correct"].mean(), 4))

    if len(parsed_df):
        print("Accuracy parsed only:", round(parsed_df["is_correct"].mean(), 4))

    print("Inference time:", round(elapsed, 2), "s")

    print("\nMost common raw outputs:")
    print(df["raw_output"].astype(str).value_counts().head(20))

    print("\nPredicted:")
    print(df["parsed_prediction_label"].value_counts(dropna=False))

    print("\nConfusion:")
    print(pd.crosstab(df["label"], df["parsed_prediction_label"], rownames=["gold"], colnames=["pred"]))

if __name__ == "__main__":
    main()
