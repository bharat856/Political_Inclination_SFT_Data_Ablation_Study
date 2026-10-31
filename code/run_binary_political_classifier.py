import argparse
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROMPT_TEMPLATE = """You are an expert in political language. Classify the text below as LEFT (1) or RIGHT (2).

Use only the content of the text.
Respond with one number only:
1 = LEFT
2 = RIGHT

Text:
{text}

Answer:"""

def parse_prediction(s):
    s = str(s).strip()
    first = s[:20].lower()

    if first.startswith("1") or "1" in first[:5]:
        return 1, "left"
    if first.startswith("2") or "2" in first[:5]:
        return 2, "right"

    low = s.lower()
    if "left" in low or "liberal" in low or "progressive" in low:
        return 1, "left"
    if "right" in low or "conservative" in low:
        return 2, "right"

    return None, None

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

    print("Loading tokenizer...")
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = [PROMPT_TEMPLATE.format(text=str(t)) for t in df["text"]]

    print("Initializing vLLM...")
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
        max_tokens=4,
    )

    print("Running binary classifier...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    raw_outputs = [o.outputs[0].text.strip() for o in outputs]
    parsed = [parse_prediction(x) for x in raw_outputs]

    df["raw_output"] = raw_outputs
    df["parsed_prediction_num"] = [p[0] for p in parsed]
    df["parsed_prediction_label"] = [p[1] for p in parsed]
    df["parse_ok"] = df["parsed_prediction_label"].notna()
    df["is_correct"] = df["parsed_prediction_label"] == df["label"]

    out_path = out_dir / f"binary_classifier__{safe_model_name(args.model)}__{args.split}.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Parsed:", int(df["parse_ok"].sum()), "/", len(df))
    print("Accuracy:", round(df["is_correct"].mean(), 4))
    print("Inference time:", round(elapsed, 2), "s")

    print("\nPredicted:")
    print(df["parsed_prediction_label"].value_counts())

    print("\nConfusion:")
    print(pd.crosstab(df["label"], df["parsed_prediction_label"], rownames=["gold"], colnames=["pred"]))

if __name__ == "__main__":
    main()
