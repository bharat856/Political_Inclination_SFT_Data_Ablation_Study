from pathlib import Path
import argparse
import re
import time
import pandas as pd

from vllm import LLM, SamplingParams


SYSTEM_PROMPT = """### PROMPT:
You are an expert in political language. Classify the document below as LEFT (1), NEUTRAL (2), or RIGHT (3) based on:
- **Language** (partisan terms)
- **Position** (alignment with progressive or conservative policies)
- **Framing** (balance vs. one-sidedness)
Use only the content of the document. Ignore source or author.
Respond with one number: 1 (LEFT), 2 (NEUTRAL), or 3 (RIGHT).
### DOCUMENT:
{document}
### ANSWER:"""


def parse_label(raw):
    if raw is None:
        return None, None

    s = str(raw).strip()
    m = re.search(r"\b([123])\b", s)

    if not m:
        return None, None

    label_id = int(m.group(1))
    label_map = {
        1: "left",
        2: "center_neutral",
        3: "right",
    }
    return label_id, label_map[label_id]


def build_prompts(texts, max_chars):
    prompts = []
    for t in texts:
        t = str(t)
        if max_chars and len(t) > max_chars:
            t = t[:max_chars]
        prompts.append(SYSTEM_PROMPT.format(document=t))
    return prompts


def run_model(model_name, model_path, tp, pp, df, text_col, output_dir, mode, max_chars):
    print("=" * 100)
    print(f"Running model: {model_name}")
    print(f"Path: {model_path}")
    print(f"Mode: {mode}")
    print(f"Rows: {len(df)}")
    print("=" * 100)

    prompts = build_prompts(df[text_col].tolist(), max_chars=max_chars)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    raw_outputs = []
    label_ids = []
    labels = []

    for out in outputs:
        raw = out.outputs[0].text.strip() if out.outputs else ""
        label_id, label = parse_label(raw)
        raw_outputs.append(raw)
        label_ids.append(label_id)
        labels.append(label)

    result = df.copy()
    result[f"{model_name}_raw_output"] = raw_outputs
    result[f"{model_name}_label_id"] = label_ids
    result[f"{model_name}_label"] = labels

    safe_name = model_name.replace("/", "_").replace(" ", "_")
    out_path = output_dir / f"{safe_name}_{mode}_original_3way_prompt.csv"
    result.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Parsed: {pd.Series(labels).notna().sum()} / {len(labels)}")
    print(f"Inference time: {elapsed:.2f} sec")
    print("\nLabel distribution:")
    print(pd.Series(labels, name=f"{model_name}_label").value_counts(dropna=False))
    print("\nRaw output examples:")
    print(pd.Series(raw_outputs).value_counts(dropna=False).head(20))

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="text_preview")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Format: name|path|tensor_parallel_size|pipeline_parallel_size"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    if args.text_column not in df.columns:
        raise ValueError(f"Text column {args.text_column} not found. Columns: {df.columns.tolist()}")

    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=42).reset_index(drop=True)

    print("Input rows:", len(df))
    print("Columns:", df.columns.tolist())

    # Mode A: original prompt directly on full input pool
    direct_df = df.copy()

    # Mode B: ModernBERT filter first, then same original 3-way prompt
    if "binary_pred_label" not in df.columns:
        raise ValueError(
            "binary_pred_label column not found. Use the ModernBERT prediction CSV as input."
        )

    filtered_df = df[df["binary_pred_label"] == "political"].copy().reset_index(drop=True)

    print("\nMode A direct rows:", len(direct_df))
    print("Mode B ModernBERT-filtered rows:", len(filtered_df))

    model_specs = []
    for spec in args.model:
        parts = spec.split("|")
        if len(parts) != 4:
            raise ValueError(
                f"Bad model spec: {spec}\nExpected: name|path|tp|pp"
            )
        name, path, tp, pp = parts
        model_specs.append((name, path, int(tp), int(pp)))

    saved_files = []

    for name, path, tp, pp in model_specs:
        saved_files.append(
            run_model(
                model_name=name,
                model_path=path,
                tp=tp,
                pp=pp,
                df=direct_df,
                text_col=args.text_column,
                output_dir=output_dir,
                mode="direct_full_pool",
                max_chars=args.max_chars,
            )
        )

        saved_files.append(
            run_model(
                model_name=name,
                model_path=path,
                tp=tp,
                pp=pp,
                df=filtered_df,
                text_col=args.text_column,
                output_dir=output_dir,
                mode="modernbert_filtered",
                max_chars=args.max_chars,
            )
        )

    print("\nAll saved files:")
    for p in saved_files:
        print(p)


if __name__ == "__main__":
    main()
