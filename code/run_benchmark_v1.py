#!/usr/bin/env python3
"""Run benchmark_v1 with vLLM or Transformers."""

from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_prompts import build_prompt
from benchmark_utils import (
    parse_prediction,
    resolve_project_path,
    safe_model_name,
    timestamp_utc,
    write_json,
)


DEFAULT_BENCHMARK = "data/processed/benchmark_v1/benchmark_v1.csv"
DEFAULT_MODELS = [
    "cyankiwi/gemma-4-31B-it-AWQ-8bit",
    "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8",
    "meta-llama/Llama-3.1-8B-Instruct",
]
DEFAULT_OUTPUT_DIR = "outputs/benchmark_v1"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Manifesto benchmark v1.")
    parser.add_argument("--benchmark", type=Path, default=Path(DEFAULT_BENCHMARK))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-transformers", type=parse_bool, default=False)
    parser.add_argument(
        "--quantization",
        default=None,
        help=(
            "Optional vLLM quantization hint, e.g. awq. Leave unset to let "
            "vLLM use model config metadata."
        ),
    )
    return parser.parse_args()


def prediction_path(output_dir: Path, model_name: str, prompt_version: str) -> Path:
    return output_dir / "predictions" / f"{safe_model_name(model_name)}__prompt_{prompt_version}.csv"


def load_benchmark(path: Path, limit: int | None) -> pd.DataFrame:
    print(f"Loading benchmark: {path}", flush=True)
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit).copy()
    print(f"Rows to run: {len(df)}", flush=True)
    if "text" not in df.columns or "gold_label" not in df.columns:
        raise ValueError("Benchmark must contain text and gold_label columns.")
    return df


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_name}", flush=True)
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def make_chat_prompt(tokenizer: Any, text: Any, prompt_version: str) -> str:
    content = build_prompt(text, prompt_version=prompt_version)
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def run_vllm(
    model_name: str,
    tokenizer: Any,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[str], float]:
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM model: {model_name}", flush=True)
    load_start = time.perf_counter()
    print(f"vLLM quantization override: {args.quantization}", flush=True)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        quantization=args.quantization if args.quantization else None,
    )
    print(f"vLLM initialized in {time.perf_counter() - load_start:.2f}s", flush=True)

    prompts = [make_chat_prompt(tokenizer, text, args.prompt_version) for text in df["text"]]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    infer_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - infer_start
    raw_outputs = [output.outputs[0].text.strip() for output in outputs]
    del llm
    return raw_outputs, elapsed


def generate_transformers(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> str:
    input_device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    outputs = model.generate(**inputs, **generation_kwargs)
    new_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_transformers(
    model_name: str,
    tokenizer: Any,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[str], float]:
    import torch
    from transformers import AutoModelForCausalLM

    print(f"Initializing Transformers model: {model_name}", flush=True)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Transformers initialized in {time.perf_counter() - load_start:.2f}s", flush=True)

    raw_outputs: list[str] = []
    infer_start = time.perf_counter()
    for row_number, text in enumerate(df["text"], start=1):
        print(f"Transformers row {row_number}/{len(df)}", flush=True)
        prompt = make_chat_prompt(tokenizer, text, args.prompt_version)
        raw_outputs.append(
            generate_transformers(model, tokenizer, prompt, args.max_tokens, args.temperature)
        )
    elapsed = time.perf_counter() - infer_start
    del model
    torch.cuda.empty_cache()
    return raw_outputs, elapsed


def add_prediction_columns(
    df: pd.DataFrame,
    model_name: str,
    backend: str,
    prompt_version: str,
    raw_outputs: list[str],
    elapsed: float,
) -> pd.DataFrame:
    result = df.copy()
    per_row_time = elapsed / len(result) if len(result) else 0.0
    parsed = [parse_prediction(raw) for raw in raw_outputs]
    result["model_name"] = model_name
    result["backend"] = backend
    result["prompt_version"] = prompt_version
    result["raw_output"] = raw_outputs
    result["parsed_prediction_num"] = [num for num, _, _ in parsed]
    result["parsed_prediction_label"] = [label for _, label, _ in parsed]
    result["parse_ok"] = [ok for _, _, ok in parsed]
    result["is_correct"] = result["parsed_prediction_label"] == result["gold_label"]
    result.loc[~result["parse_ok"], "is_correct"] = False
    result["inference_time_sec"] = per_row_time
    return result


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_one_model(model_name: str, df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    out_path = prediction_path(output_dir, model_name, args.prompt_version)
    if args.resume and out_path.exists():
        existing = pd.read_csv(out_path)
        if len(existing) >= len(df):
            print(f"Skipping existing complete predictions: {out_path}", flush=True)
            return

    errors_dir = output_dir / "errors"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    attempted_backend = args.backend
    try:
        tokenizer = load_tokenizer(model_name)
        if args.backend == "vllm":
            try:
                raw_outputs, elapsed = run_vllm(model_name, tokenizer, df, args)
            except Exception:
                if not args.fallback_transformers:
                    raise
                print(f"vLLM failed for {model_name}; falling back to Transformers.", flush=True)
                attempted_backend = "transformers"
                cleanup_cuda()
                raw_outputs, elapsed = run_transformers(model_name, tokenizer, df, args)
        else:
            raw_outputs, elapsed = run_transformers(model_name, tokenizer, df, args)

        result = add_prediction_columns(
            df=df,
            model_name=model_name,
            backend=attempted_backend,
            prompt_version=args.prompt_version,
            raw_outputs=raw_outputs,
            elapsed=elapsed,
        )
        result.to_csv(out_path, index=False)
        parsed_count = int(result["parse_ok"].sum())
        accuracy = float(result["is_correct"].mean()) if len(result) else 0.0
        print(f"Saved predictions: {out_path}", flush=True)
        print(f"Parsed rows: {parsed_count}/{len(result)} accuracy={accuracy:.4f}", flush=True)
    except Exception as exc:
        error_path = errors_dir / f"{safe_model_name(model_name)}.json"
        write_json(
            error_path,
            {
                "model_name": model_name,
                "backend": attempted_backend,
                "prompt_version": args.prompt_version,
                "timestamp": timestamp_utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"Model failed, wrote error JSON: {error_path}", flush=True)
    finally:
        cleanup_cuda()


def main() -> None:
    args = parse_args()
    benchmark_path = resolve_project_path(args.benchmark)
    output_dir = resolve_project_path(args.output_dir)
    df = load_benchmark(benchmark_path, args.limit)
    print(f"Models: {args.models}", flush=True)
    print(f"Backend: {args.backend}", flush=True)
    print(f"Prompt version: {args.prompt_version}", flush=True)

    for model_number, model_name in enumerate(args.models, start=1):
        print(f"Starting model {model_number}/{len(args.models)}: {model_name}", flush=True)
        run_one_model(model_name, df, args, output_dir)


if __name__ == "__main__":
    main()
