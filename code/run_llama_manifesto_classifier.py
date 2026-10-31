#!/usr/bin/env python3
"""Run a political leaning classifier on Manifesto chunk benchmarks."""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path("/fs/scratch/users/bharatpi/causal_political_content")
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/manifesto/chunks/sample_30_manifesto_chunks_strict_balanced.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs/classifier/llama_manifesto_predictions.csv"
)
DEFAULT_MODEL_PATH = (
    "hf_cache/models--unsloth--Llama-3.3-70B-Instruct-bnb-4bit/"
    "snapshots/75779cdfa1240a4d048c6ecf65e97ff31b707214"
)

SYSTEM_PROMPT = """You are an expert in political language. Classify the document below as LEFT (1), CENTER/NEUTRAL (2), or RIGHT (3) based on:
- political position
- policy framing
- ideological language
Use only the content of the document. Ignore party name, country, source, and author.
Respond with one number only: 1, 2, or 3."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Manifesto chunks as left, center/neutral, or right "
            "with vLLM or Transformers."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input Manifesto chunk CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Model path or Hugging Face ID. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers"],
        default="vllm",
        help="Inference backend to use. Default: vllm",
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="Convenience alias for --backend transformers.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to classify for a quick test run.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor or pipeline parallel size. Keep this at 2 or less.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory vLLM may use.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Run vLLM with eager execution enabled.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5,
        help="Maximum generated tokens per classification.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load CSV/tokenizer, print the first formatted prompt, and exit before model init.",
    )
    parser.add_argument(
        "--single-prompt-test",
        action="store_true",
        help="Load the selected backend, run one tiny hardcoded prompt, print output, and exit.",
    )
    parser.add_argument(
        "--quiet-vllm",
        action="store_true",
        default=False,
        help="Reduce vLLM, transformers, and Hugging Face Hub logging noise.",
    )
    parser.add_argument(
        "--suppress-transformers-warnings",
        action="store_true",
        default=True,
        help="Suppress non-critical transformers compatibility/deprecation warnings.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.use_transformers:
        args.backend = "transformers"
    if args.tensor_parallel_size > 2:
        raise ValueError("--tensor-parallel-size must be 2 or less for this project.")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be at least 1.")
    if args.max_model_len < 1:
        raise ValueError("--max-model-len must be positive.")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive.")


def configure_logging(args: argparse.Namespace) -> None:
    if args.suppress_transformers_warnings:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings(
            "ignore", message=".*Unrecognized keys in `rope_parameters`.*"
        )
        warnings.filterwarnings("ignore", message=".*Qwen2VLImageProcessorFast.*")
        warnings.filterwarnings("ignore", message=".*use_fast.*deprecated.*")

    if args.quiet_vllm:
        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        logging.getLogger("vllm").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def resolve_model_path(model: str) -> str:
    model_path = Path(model)
    if model_path.is_absolute():
        return str(model_path)

    project_relative = PROJECT_ROOT / model_path
    if project_relative.exists():
        return str(project_relative)

    return model


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_prompt(tokenizer: Any, text: Any) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### DOCUMENT:\n{str(text)}\n### ANSWER:"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_prediction(raw_prediction: str) -> tuple[str | None, str]:
    cleaned = raw_prediction.strip()
    if cleaned.startswith("1"):
        return "1", "left"
    if cleaned.startswith("2"):
        return "2", "center"
    if cleaned.startswith("3"):
        return "3", "right"
    return None, "unknown"


def normalize_label(label: Any) -> str:
    return str(label).strip().lower()


def make_llm(model_path: str, args: argparse.Namespace) -> Any:
    from vllm import LLM

    model_lower = model_path.lower()
    if "bnb" in model_lower:
        return LLM(
            model=model_path,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            max_model_len=args.max_model_len,
            pipeline_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )

    return LLM(
        model=model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )


def make_transformers_model(model_path: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )


def get_transformers_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def classify_row_vllm(
    llm: Any,
    tokenizer: Any,
    sampling_params: Any,
    text: Any,
) -> str:
    prompt = build_prompt(tokenizer, text)
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def classify_row_transformers(
    model: Any,
    tokenizer: Any,
    text: Any,
    max_tokens: int,
) -> str:
    prompt = build_prompt(tokenizer, text)
    return generate_transformers(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    )


def generate_transformers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
) -> str:
    print("Running Transformers generation...", flush=True)
    input_device = get_transformers_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_input_csv(input_path: Path, limit: int | None) -> pd.DataFrame:
    print("Loading input CSV...", flush=True)
    df = pd.read_csv(input_path)
    if limit is not None:
        df = df.head(limit).copy()
    else:
        df = df.copy()
    print(f"Number of rows: {len(df)}", flush=True)
    return df


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    print("Loading tokenizer...", flush=True)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elapsed = time.perf_counter() - start
    print("Tokenizer loaded.", flush=True)
    print(f"Tokenizer loading time: {elapsed:.2f} seconds", flush=True)
    return tokenizer


def print_model_loading_warning(model_path: str) -> None:
    print(f"Selected model path: {model_path}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"HF_HOME: {os.environ.get('HF_HOME', '')}", flush=True)
    print(f"VLLM_CACHE_ROOT: {os.environ.get('VLLM_CACHE_ROOT', '')}", flush=True)
    print(f"VLLM_CACHE_DIR: {os.environ.get('VLLM_CACHE_DIR', '')}", flush=True)
    print("If this hangs, check nvidia-smi in another terminal.", flush=True)


def initialize_llm(model_path: str, args: argparse.Namespace) -> Any:
    print_model_loading_warning(model_path)
    print("Initializing vLLM model...", flush=True)
    start = time.perf_counter()
    llm = make_llm(model_path, args)
    elapsed = time.perf_counter() - start
    print("vLLM model initialized.", flush=True)
    print(f"vLLM initialization time: {elapsed:.2f} seconds", flush=True)
    return llm


def initialize_transformers_model(model_path: str) -> Any:
    print_model_loading_warning(model_path)
    print("Initializing Transformers model...", flush=True)
    start = time.perf_counter()
    model = make_transformers_model(model_path)
    elapsed = time.perf_counter() - start
    print("Transformers model initialized.", flush=True)
    print(f"Transformers initialization time: {elapsed:.2f} seconds", flush=True)
    return model


def run_dry_run(df: pd.DataFrame, tokenizer: Any) -> None:
    if df.empty:
        print("Dry run skipped prompt build because the input CSV has no rows.", flush=True)
        return

    prompt = build_prompt(tokenizer, df.iloc[0]["text"])
    print("First formatted prompt:", flush=True)
    print(prompt, flush=True)


def run_single_prompt_test(
    model_path: str,
    args: argparse.Namespace,
    tokenizer: Any | None = None,
) -> None:
    prompt = "Classify: Workers should receive higher wages. Answer 1, 2, or 3."

    if args.backend == "vllm":
        from vllm import SamplingParams

        llm = initialize_llm(model_path, args)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print("Starting single prompt test...", flush=True)
        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start
        raw_output = outputs[0].outputs[0].text.strip()
        print(f"Single prompt raw output: {raw_output}", flush=True)
        print(f"Single prompt inference time: {elapsed:.2f} seconds", flush=True)
        return

    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)

    model = initialize_transformers_model(model_path)
    print("Starting single prompt test...", flush=True)
    start = time.perf_counter()
    raw_output = generate_transformers(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
    )
    elapsed = time.perf_counter() - start
    print(f"Single prompt raw output: {raw_output}", flush=True)
    print(f"Single prompt inference time: {elapsed:.2f} seconds", flush=True)


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    validate_args(args)
    configure_logging(args)
    args.input = resolve_project_path(args.input)
    args.output = resolve_project_path(args.output)
    model_path = resolve_model_path(args.model)

    print(f"Input path: {args.input}", flush=True)
    print(f"Output path: {args.output}", flush=True)
    print(f"Model path: {model_path}", flush=True)
    print(f"Backend: {args.backend}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
    print(f"HF_HOME: {os.environ.get('HF_HOME', '')}", flush=True)
    print(f"VLLM_CACHE_ROOT: {os.environ.get('VLLM_CACHE_ROOT', '')}", flush=True)
    print(f"VLLM_CACHE_DIR: {os.environ.get('VLLM_CACHE_DIR', '')}", flush=True)

    if args.single_prompt_test:
        run_single_prompt_test(model_path, args)
        total_elapsed = time.perf_counter() - total_start
        print(f"Total script time: {total_elapsed:.2f} seconds", flush=True)
        return

    df = load_input_csv(args.input, args.limit)
    tokenizer = load_tokenizer(model_path)

    if args.dry_run:
        run_dry_run(df, tokenizer)
        total_elapsed = time.perf_counter() - total_start
        print(f"Total script time: {total_elapsed:.2f} seconds", flush=True)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "vllm":
        from vllm import SamplingParams

        backend_model = initialize_llm(model_path, args)
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        backend_model = initialize_transformers_model(model_path)
        sampling_params = None

    has_labels = "rile_label_3way" in df.columns

    df["model_name"] = model_path
    df["raw_prediction"] = None
    df["parsed_prediction_num"] = None
    df["parsed_prediction_label"] = None
    df["is_correct"] = None

    print("Starting inference...", flush=True)

    for row_number, (idx, row) in enumerate(df.iterrows(), start=1):
        chunk_id = row.get("chunk_id", idx)
        print(f"Running row {row_number}/{len(df)}: {chunk_id}", flush=True)
        row_start = time.perf_counter()
        try:
            if args.backend == "vllm":
                raw_prediction = classify_row_vllm(
                    llm=backend_model,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    text=row["text"],
                )
            else:
                raw_prediction = classify_row_transformers(
                    model=backend_model,
                    tokenizer=tokenizer,
                    text=row["text"],
                    max_tokens=args.max_tokens,
                )
        except Exception as exc:
            raw_prediction = "ERROR"
            print(f"Row {row_number}/{len(df)} failed: {exc}", flush=True)

        parsed_num, parsed_label = parse_prediction(raw_prediction)

        if has_labels:
            gold_label = normalize_label(row["rile_label_3way"])
            is_correct = parsed_label == gold_label
        else:
            is_correct = None

        df.at[idx, "raw_prediction"] = raw_prediction
        df.at[idx, "parsed_prediction_num"] = parsed_num
        df.at[idx, "parsed_prediction_label"] = parsed_label
        df.at[idx, "is_correct"] = is_correct

        df.to_csv(args.output, index=False)

        row_elapsed = time.perf_counter() - row_start
        print(
            f"Finished row {row_number}/{len(df)}: raw_prediction={raw_prediction}",
            flush=True,
        )
        print(f"Row inference time: {row_elapsed:.2f} seconds", flush=True)

    df.to_csv(args.output, index=False)

    print(f"Saved predictions: {args.output}", flush=True)
    if has_labels and len(df) > 0:
        accuracy = df["is_correct"].mean()
        parsed = df["parsed_prediction_label"] != "unknown"
        print(f"Final accuracy: {accuracy:.4f}", flush=True)
        print(f"Parsed rows: {int(parsed.sum())}/{len(df)}", flush=True)

    total_elapsed = time.perf_counter() - total_start
    print(f"Total script time: {total_elapsed:.2f} seconds", flush=True)


if __name__ == "__main__":
    main()

# Command examples:
# source scripts/setup_env.sh
# python code/run_llama_manifesto_classifier.py --dry-run --limit 1
# python code/run_llama_manifesto_classifier.py --single-prompt-test --model <MODEL_PATH>
# python code/run_llama_manifesto_classifier.py --backend transformers --single-prompt-test --model <MODEL_PATH>
# python code/run_llama_manifesto_classifier.py --model <MODEL_PATH> --limit 5 --output outputs/classifier/qwen_test_5_predictions.csv
