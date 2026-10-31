import re
import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer



PROMPT_TEMPLATE = """### PROMPT:
You are an expert in political language. Classify the document below as LEFT (1), NEUTRAL (2), or RIGHT (3) based on:
- **Language** (partisan terms)
- **Position** (alignment with progressive or conservative policies)
- **Framing** (balance vs. one-sidedness)

Use only the content of the document. Ignore source or author.
Respond with one number only: 1, 2, or 3.

### DOCUMENT:
{text}

### ANSWER:"""


def parse_prediction(txt):
    """
    Parse Dmitry-style 3-way output:
    1 = left
    2 = center_neutral
    3 = right
    """
    if txt is None:
        return None
    txt = str(txt).strip()
    m = re.search(r"([1-3])", txt)
    if m:
        return int(m.group(1))
    return None

def truncate_text(text, max_chars=6000):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-column", default="full_text")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_prompt_tokens = args.max_model_len - 16

    prefix, suffix = PROMPT_TEMPLATE.split("{text}")

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    available_text_tokens = max_prompt_tokens - len(prefix_tokens) - len(suffix_tokens)

    prompts = []
    truncated_by_tokens = 0

    for x in df[args.text_column].fillna(""):
        text = truncate_text(x, args.max_chars)
        prompt = PROMPT_TEMPLATE.format(text=text)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if len(prompt_ids) > max_prompt_tokens:
            text_ids = tokenizer.encode(str(x), add_special_tokens=False)
            text_ids = text_ids[:max(1, available_text_tokens)]
            text = tokenizer.decode(text_ids, skip_special_tokens=True) + "\n\n[TRUNCATED_BY_TOKENS]"
            prompt = PROMPT_TEMPLATE.format(text=text)
            truncated_by_tokens += 1

        prompts.append(prompt)

    print("Token-truncated prompts:", truncated_by_tokens)

    print("Loading model:", args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=8,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling)
    elapsed = time.time() - start

    raw_outputs = []
    parsed = []

    for out in outputs:
        txt = out.outputs[0].text.strip() if out.outputs else ""
        raw_outputs.append(txt)
        parsed.append(parse_prediction(txt))

    df["sft_political_raw_output"] = raw_outputs
    df["sft_political_label_id"] = parsed

    label_map = {
        0: "non_political",
        1: "left",
        2: "center_neutral",
        3: "right",
    }

    df["sft_political_label"] = df["sft_political_label_id"].map(label_map)
    df["parse_ok"] = df["sft_political_label_id"].notna()

    model_name = args.model.replace("/", "_").replace(".", "_").replace("-", "_")
    out_file = out_dir / f"sft_political_4way__{model_name}.csv"
    df.to_csv(out_file, index=False)

    print("Saved:", out_file)
    print("Parsed:", int(df["parse_ok"].sum()), "/", len(df))
    print("Inference time:", round(elapsed, 2), "s")

    print("\nLabel distribution:")
    print(df["sft_political_label"].value_counts(dropna=False))

    print("\nBy source:")
    print(pd.crosstab(df["source"], df["sft_political_label"], dropna=False).head(50))

    print("\nRaw output examples:")
    print(df["sft_political_raw_output"].value_counts(dropna=False).head(20))


if __name__ == "__main__":
    main()
