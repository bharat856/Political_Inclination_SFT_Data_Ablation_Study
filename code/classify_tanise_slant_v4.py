from pathlib import Path
import argparse
import json
import os
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


LABEL_TO_NUM = {
    "left": 1,
    "neutral": 2,
    "center": 2,
    "right": 3,
}

NUM_TO_LABEL = {
    1: "left",
    2: "neutral",
    3: "right",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_tag")
    parser.add_argument(
        "--data-path",
        default="data/validation_tanise/slant_testset_for_4way.csv",
    )
    parser.add_argument(
        "--prompt-path",
        default="zero_4_v4.prompt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def parse_answer(text):
    match = re.search(r"\b([123])\b", str(text).strip())
    return int(match.group(1)) if match else None


def load_existing_ids(path):
    if not path.exists():
        return set()

    try:
        existing = pd.read_csv(path, usecols=["validation_id"])
    except Exception:
        return set()

    return set(existing["validation_id"].astype(str))


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    prompt_path = Path(args.prompt_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if args.text_column not in df.columns:
        raise KeyError(
            f"Missing text column {args.text_column!r}. "
            f"Columns: {df.columns.tolist()}"
        )

    if args.label_column not in df.columns:
        raise KeyError(
            f"Missing label column {args.label_column!r}. "
            f"Columns: {df.columns.tolist()}"
        )

    if args.limit is not None:
        df = df.head(args.limit).copy()

    df = df.reset_index(drop=False).rename(
        columns={"index": "source_row_index"}
    )

    # Stable validation identifier independent of output line order.
    if "validation_id" not in df.columns:
        df["validation_id"] = [
            f"tanise_slant_{i:04d}"
            for i in range(len(df))
        ]

    prompt_template = prompt_path.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_tag,
        trust_remote_code=True,
    )

    num_gpus = torch.cuda.device_count()

    llm_kwargs = {
        "model": args.model_tag,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.75,
        "max_model_len": args.max_model_len,
    }

    if num_gpus > 1:
        llm_kwargs["tensor_parallel_size"] = num_gpus
        llm_kwargs["disable_custom_all_reduce"] = True



    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    completed_ids = load_existing_ids(output_path)
    append = output_path.exists() and output_path.stat().st_size > 0

    rows_to_process = df[
        ~df["validation_id"].astype(str).isin(completed_ids)
    ].copy()

    print("Model:", args.model_tag)
    print("Input rows:", len(df))
    print("Already completed:", len(completed_ids))
    print("Rows remaining:", len(rows_to_process))
    print("Output:", output_path)

    if rows_to_process.empty:
        print("Nothing to do.")
        return

    records = rows_to_process.to_dict(orient="records")

    with output_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as out_handle:
        first_write = not append

        for start in tqdm(
            range(0, len(records), args.batch_size),
            desc="classification batches",
        ):
            batch = records[start:start + args.batch_size]
            chat_inputs = []

            for row in batch:
                prompt = prompt_template.format(
                    content=str(row[args.text_column])
                )

                messages = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]

                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }

                if "Qwen3.5" in args.model_tag:
                    kwargs["enable_thinking"] = False

                chat_inputs.append(
                    tokenizer.apply_chat_template(
                        messages,
                        **kwargs,
                    )
                )

            outputs = llm.generate(
                chat_inputs,
                sampling_params,
                use_tqdm=False,
            )

            result_rows = []

            for row, output in zip(batch, outputs):
                raw = output.outputs[0].text.strip()
                pred_num = parse_answer(raw)
                pred_label = (
                    NUM_TO_LABEL.get(pred_num)
                    if pred_num is not None
                    else None
                )

                gold_label = str(
                    row[args.label_column]
                ).strip().lower()

                gold_num = LABEL_TO_NUM.get(gold_label)

                result_rows.append({
                    "validation_id": row["validation_id"],
                    "source_row_index": row["source_row_index"],
                    "model_name": args.model_tag,
                    "text": row[args.text_column],
                    "gold_label": gold_label,
                    "gold_label_num": gold_num,
                    "raw_output": raw,
                    "parsed_prediction_num": pred_num,
                    "parsed_prediction_label": pred_label,
                    "parse_ok": pred_num is not None,
                    "is_correct": (
                        pred_num == gold_num
                        if pred_num is not None
                        and gold_num is not None
                        else False
                    ),
                })

            result_df = pd.DataFrame(result_rows)
            result_df.to_csv(
                out_handle,
                index=False,
                header=first_write,
            )

            out_handle.flush()
            first_write = False

    metadata = {
        "model": args.model_tag,
        "data_path": str(data_path),
        "prompt_path": str(prompt_path),
        "output": str(output_path),
        "rows": len(df),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
    }

    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Saved:", output_path)


if __name__ == "__main__":
    main()
