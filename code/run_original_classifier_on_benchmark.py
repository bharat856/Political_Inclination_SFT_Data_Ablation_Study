import argparse
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """### PROMPT:
You are an expert in political language. Classify the document below as LEFT (1), NEUTRAL (2), or RIGHT (3) based on:
- **Language** (partisan terms)
- **Position** (alignment with progressive or conservative policies)
- **Framing** (balance vs. one-sidedness)
Use only the content of the document. Ignore source or author.
Respond with one number: 1 (LEFT), 2 (NEUTRAL), or 3 (RIGHT).
### DOCUMENT: 
### ANSWER:"""


def parse_prediction(raw_text: str):
    """
    Extract first 1/2/3 from model output.
    """
    if raw_text is None:
        return None

    text = str(raw_text).strip()

    match = re.search(r"\b([123])\b", text)
    if match:
        return int(match.group(1))

    # fallback for outputs like "LEFT", "NEUTRAL", "RIGHT"
    lowered = text.lower()
    if "left" in lowered:
        return 1
    if "neutral" in lowered or "center" in lowered or "centre" in lowered:
        return 2
    if "right" in lowered:
        return 3

    return None


def num_to_label(x):
    return {1: "left", 2: "center", 3: "right"}.get(x, None)


def build_prompt(tokenizer, text: str):
    """
    This follows the original classifier style:
    user_prompt = '### DOCUMENT:\\n' + text + '\\n### ANSWER'
    messages = system + user
    apply_chat_template(...)
    """
    user_prompt = "### DOCUMENT:\n" + str(text) + "\n### ANSWER"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/original_classifier")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark: {benchmark_path}")
    df = pd.read_csv(benchmark_path)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Rows to run: {len(df)}")
    print(f"Model: {args.model}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )

    print("Building prompts...")
    prompts = [build_prompt(tokenizer, text) for text in df["text"].tolist()]

    is_bnb = "bnb" in args.model.lower() or "bitsandbytes" in args.model.lower()

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
    )

    if is_bnb:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"

    print("Initializing vLLM...")
    start_init = time.time()
    llm = LLM(**llm_kwargs)
    print(f"vLLM initialized in {time.time() - start_init:.2f}s")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("Running original classifier...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    raw_outputs = [o.outputs[0].text.strip() for o in outputs]
    parsed_nums = [parse_prediction(x) for x in raw_outputs]
    parsed_labels = [num_to_label(x) for x in parsed_nums]

    df["original_classifier_raw_output"] = raw_outputs
    df["parsed_prediction_num"] = parsed_nums
    df["parsed_prediction_label"] = parsed_labels
    df["parse_ok"] = df["parsed_prediction_num"].notna()

    if "gold_label_num" in df.columns:
        gold_num = df["gold_label_num"]
    else:
        gold_num = df["rile_label_3way"].map({"left": 1, "center": 2, "right": 3})

    df["gold_label_num_eval"] = gold_num
    df["is_correct"] = df["parsed_prediction_num"] == df["gold_label_num_eval"]

    safe_model_name = (
        args.model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "-")
        .replace(".", ".")
    )
    if safe_model_name.startswith("_"):
        safe_model_name = safe_model_name[1:]

    pred_path = output_dir / f"original_classifier__{safe_model_name}.csv"
    df.to_csv(pred_path, index=False)

    accuracy = df["is_correct"].mean()
    parsed_count = int(df["parse_ok"].sum())

    print(f"Saved predictions: {pred_path}")
    print(f"Parsed rows: {parsed_count}/{len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Inference time: {elapsed:.2f}s")

    print("\nGold label distribution:")
    print(df["rile_label_3way"].value_counts(dropna=False))

    print("\nPredicted label distribution:")
    print(df["parsed_prediction_label"].value_counts(dropna=False))

    print("\nConfusion table:")
    print(pd.crosstab(
        df["rile_label_3way"],
        df["parsed_prediction_label"],
        rownames=["gold"],
        colnames=["pred"],
        dropna=False,
    ))


if __name__ == "__main__":
    main()
