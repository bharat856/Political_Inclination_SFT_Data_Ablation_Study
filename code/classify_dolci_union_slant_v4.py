from pathlib import Path
import argparse
import json
import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


NUM_TO_LABEL = {
    1: "left",
    2: "neutral",
    3: "right",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("model_tag")

    p.add_argument(
        "--input",
        default=(
            "data/interventions/"
            "dolci_olmo3_sft_union_mpnet_t04/"
            "all_union_candidates_for_3way.jsonl"
        ),
    )

    p.add_argument(
        "--prompt-path",
        default="zero_4_v4.prompt",
    )

    p.add_argument("--output", required=True)

    p.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
    )

    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
    )

    return p.parse_args()


def parse_answer(text):
    match = re.search(
        r"\b([123])\b",
        str(text).strip(),
    )

    if match:
        return int(match.group(1))

    return None


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_lines(path):
    if not path.exists():
        return 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        return sum(1 for line in f if line.strip())


def get_text(obj):
    for field in [
        "text",
        "content",
        "full_text",
    ]:
        value = obj.get(field)

        if value is not None and str(value).strip():
            return str(value)

    return ""


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_template = Path(
        args.prompt_path
    ).read_text(
        encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_tag,
        trust_remote_code=True,
    )

    num_gpus = torch.cuda.device_count()

    llm_kwargs = {
        "model": args.model_tag,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
        ),
    }

    if num_gpus > 1:
        # bnb Llama models have previously been run with
        # pipeline parallelism in this project.
        if "bnb" in args.model_tag.lower():
            llm_kwargs[
                "pipeline_parallel_size"
            ] = num_gpus
        else:
            llm_kwargs[
                "tensor_parallel_size"
            ] = num_gpus
            llm_kwargs[
                "disable_custom_all_reduce"
            ] = True

    print("Model:", args.model_tag)
    print("Visible GPUs:", num_gpus)
    print("LLM args:", llm_kwargs)

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_output_tokens,
    )

    already_done = count_lines(output_path)

    print("Existing output rows:", already_done)

    # Leave enough room for chat template + instructions + output.
    max_input_tokens = (
        args.max_model_len
        - args.max_output_tokens
        - 512
    )

    def make_chat_input(text):
        # Token-level truncation is essential because DOLCI
        # contains some very long examples.
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        truncated = len(tokens) > max_input_tokens

        if truncated:
            tokens = tokens[:max_input_tokens]
            text = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
            )

        user_prompt = prompt_template.format(
            content=text
        )

        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if "Qwen3.5" in args.model_tag:
            kwargs["enable_thinking"] = False

        chat_input = tokenizer.apply_chat_template(
            messages,
            **kwargs,
        )

        return chat_input, truncated

    batch_records = []
    batch_prompts = []
    batch_truncated = []

    def write_batch(out):
        if not batch_records:
            return

        outputs = llm.generate(
            batch_prompts,
            sampling_params,
            use_tqdm=False,
        )

        for obj, generated, was_truncated in zip(
            batch_records,
            outputs,
            batch_truncated,
        ):
            raw = (
                generated.outputs[0]
                .text
                .strip()
            )

            pred_num = parse_answer(raw)

            result = {
                "document_id": obj["document_id"],
                "split": obj["split"],
                "row_id": int(obj["row_id"]),
                "raw_output": raw,
                "prediction_num": pred_num,
                "prediction_label": (
                    NUM_TO_LABEL.get(pred_num)
                    if pred_num is not None
                    else None
                ),
                "parse_ok": (
                    pred_num is not None
                ),
                "input_truncated": bool(
                    was_truncated
                ),
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

        out.flush()

        batch_records.clear()
        batch_prompts.clear()
        batch_truncated.clear()

    total_seen = 0
    processed_this_run = 0

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as out:
        for obj in tqdm(
            read_jsonl(input_path),
            desc="DOLCI slant classification",
        ):
            if total_seen < already_done:
                total_seen += 1
                continue

            total_seen += 1

            text = get_text(obj)

            if not text.strip():
                result = {
                    "document_id": obj["document_id"],
                    "split": obj["split"],
                    "row_id": int(obj["row_id"]),
                    "raw_output": "",
                    "prediction_num": None,
                    "prediction_label": None,
                    "parse_ok": False,
                    "input_truncated": False,
                }

                out.write(
                    json.dumps(result)
                    + "\n"
                )

                out.flush()
                continue

            chat_input, truncated = make_chat_input(
                text
            )

            batch_records.append(obj)
            batch_prompts.append(chat_input)
            batch_truncated.append(truncated)

            if len(batch_records) >= args.batch_size:
                write_batch(out)
                processed_this_run += args.batch_size

        write_batch(out)

    print("Finished.")
    print("Output:", output_path)


if __name__ == "__main__":
    main()
