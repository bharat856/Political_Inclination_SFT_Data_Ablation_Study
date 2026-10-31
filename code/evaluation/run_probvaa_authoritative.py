"""Prepare and run the pinned authoritative ProbVAA evaluation.

The CPU-safe ``prepare`` command never imports model libraries.  ``worker``
loads one model instance on the single GPU visible to that process.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Mapping

from probvaa_authoritative_pipeline import (
    AUTHORITATIVE_COMMIT,
    AUTHORITATIVE_ITERATIONS,
    AUTHORITATIVE_SEED,
    REQUESTED_GENERATION_CONFIG,
    REPOSITORY_ROOT,
    append_jsonl_record,
    effective_generation_config,
    iter_generation_tasks,
    merge_results,
    pending_tasks,
    read_jsonl,
    recorded_generation_ids,
    result_from_failure,
    result_from_response,
    summarize_results,
    write_csv_exclusive,
    write_jsonl_exclusive,
)


MODEL_SPECS = {
    "official": {
        "model_id": "allenai/Olmo-3-7B-Instruct-SFT",
        "model_path": REPOSITORY_ROOT
        / "validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT",
        "adapter_path": None,
        "prompt_mode": "chat_template",
    },
    "Full": {
        "model_id": "allenai/Olmo-3-1025-7B",
        "model_path": REPOSITORY_ROOT / "hf_cache/allenai_Olmo-3-1025-7B",
        "adapter_path": REPOSITORY_ROOT
        / "outputs/sft_runs/olmo3_instruct_full_4gpu_v4",
        "prompt_mode": "raw",
    },
    "NoPol": {
        "model_id": "allenai/Olmo-3-1025-7B",
        "model_path": REPOSITORY_ROOT / "hf_cache/allenai_Olmo-3-1025-7B",
        "adapter_path": REPOSITORY_ROOT
        / "outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5",
        "prompt_mode": "raw",
    },
    "NoIdeology": {
        "model_id": "allenai/Olmo-3-1025-7B",
        "model_path": REPOSITORY_ROOT / "hf_cache/allenai_Olmo-3-1025-7B",
        "adapter_path": REPOSITORY_ROOT
        / "outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5",
        "prompt_mode": "raw",
    },
    "NoLeft": {
        "model_id": "allenai/Olmo-3-1025-7B",
        "model_path": REPOSITORY_ROOT / "hf_cache/allenai_Olmo-3-1025-7B",
        "adapter_path": REPOSITORY_ROOT
        / "outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4",
        "prompt_mode": "raw",
    },
    "NoRight": {
        "model_id": "allenai/Olmo-3-1025-7B",
        "model_path": REPOSITORY_ROOT / "hf_cache/allenai_Olmo-3-1025-7B",
        "adapter_path": REPOSITORY_ROOT
        / "outputs/sft_runs/olmo3_instruct_noright_4gpu_v5",
        "prompt_mode": "raw",
    },
    "LRBalanced": {
        "model_id": "allenai/Olmo-3-1025-7B",
        "model_path": REPOSITORY_ROOT / "hf_cache/allenai_Olmo-3-1025-7B",
        "adapter_path": REPOSITORY_ROOT
        / "outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5",
        "prompt_mode": "raw",
    },
}

SEED_MODES = ("reset_each_generation", "seed_once")


class SeedController:
    """Apply one of the two explicit worker seeding policies.

    ``reset_each_generation`` is the authoritative default: the task seed is
    applied immediately before every generation. ``seed_once`` applies the
    worker seed once at controller construction and never reseeds per task.
    """

    def __init__(self, seed_mode: str, worker_seed: int, set_seed):
        if seed_mode not in SEED_MODES:
            raise ValueError(f"Unsupported seed mode: {seed_mode!r}")
        self.seed_mode = seed_mode
        self.worker_seed = worker_seed
        self._set_seed = set_seed
        if self.seed_mode == "seed_once":
            self._set_seed(self.worker_seed)

    def before_generation(self, task_seed: int) -> None:
        if self.seed_mode == "reset_each_generation":
            self._set_seed(task_seed)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def validate_model_spec(model_key: str) -> dict[str, object]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model key: {model_key}")
    spec = dict(MODEL_SPECS[model_key])
    model_path = Path(spec["model_path"])
    required = [
        model_path / "config.json",
        model_path / "tokenizer_config.json",
        model_path / "model.safetensors.index.json",
    ]
    if spec["prompt_mode"] == "chat_template":
        required.append(model_path / "chat_template.jinja")
    adapter_path = spec["adapter_path"]
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        required.extend(
            [adapter_path / "adapter_config.json", adapter_path / "adapter_model.safetensors"]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
    return spec


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    validate_model_spec(args.model_key)
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "tasks.jsonl"
    tasks = iter_generation_tasks(
        args.model_key,
        statement_limit=args.statement_limit,
        iterations=AUTHORITATIVE_ITERATIONS,
        newline_mode=args.newline_mode,
        world_size=args.world_size,
    )
    count = write_jsonl_exclusive(manifest_path, tasks)
    metadata = {
        "authoritative_commit": AUTHORITATIVE_COMMIT,
        "model_key": args.model_key,
        "model_spec": validate_model_spec(args.model_key),
        "statement_limit": args.statement_limit,
        "iterations": AUTHORITATIVE_ITERATIONS,
        "world_size": args.world_size,
        "newline_mode": args.newline_mode,
        "requested_generation_config": REQUESTED_GENERATION_CONFIG,
        "effective_generation_config": effective_generation_config(),
        "seed_reset_for_every_iteration": AUTHORITATIVE_SEED,
        "expected_generations": count,
        "manifest": str(manifest_path),
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, sort_keys=True, default=_json_default))


class LocalHFGenerator:
    """One-process/one-GPU model wrapper matching authoritative HF inference."""

    def __init__(
        self,
        model_key: str,
        *,
        seed_mode: str = "reset_each_generation",
        worker_seed: int = AUTHORITATIVE_SEED,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "GPU worker requires torch and transformers in the active environment"
            ) from exc
        self.torch = torch
        self.spec = validate_model_spec(model_key)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Each worker requires exactly one visible CUDA GPU; set CUDA_VISIBLE_DEVICES"
            )
        model_path = str(self.spec["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        if self.spec["adapter_path"] is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Adapter workers require peft") from exc
            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.spec["adapter_path"]),
                local_files_only=True,
                is_trainable=False,
            )
        self.model.eval()
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.device = next(self.model.parameters()).device
        # Construct this only after model loading, so seed_once starts the
        # generation stream after any library/model initialization work.
        self.seed_controller = SeedController(seed_mode, worker_seed, set_seed)

    def _tokenize(self, prompt: str):
        if self.spec["prompt_mode"] == "chat_template":
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    def generate(self, prompt: str, seed: int) -> str:
        self.seed_controller.before_generation(seed)
        tokens = self._tokenize(prompt)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        prompt_length = tokens["input_ids"].shape[1]
        config = {
            "num_return_sequences": 1,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **effective_generation_config(),
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**tokens, **config)
        return self.tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        )

    def save_rng_checkpoint(
        self,
        path: Path,
        *,
        completed_count: int,
        last_generation_id: str,
    ) -> None:
        """Atomically persist all RNG states needed by seed_once resume."""

        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
        state = {
            "seed_mode": self.seed_controller.seed_mode,
            "worker_seed": self.seed_controller.worker_seed,
            "completed_count": completed_count,
            "last_generation_id": last_generation_id,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": self.torch.get_rng_state(),
            "torch_cuda_rng_states": self.torch.cuda.get_rng_state_all(),
        }
        self.torch.save(state, temporary)
        os.replace(temporary, path)

    def restore_rng_checkpoint(
        self,
        path: Path,
        *,
        expected_count: int,
        expected_last_generation_id: str,
    ) -> None:
        """Restore a seed_once stream, refusing mismatched output/checkpoints."""

        import numpy as np

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"seed_once resume requires RNG checkpoint matching {path}"
            )
        state = self.torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "seed_mode": self.seed_controller.seed_mode,
            "worker_seed": self.seed_controller.worker_seed,
            "completed_count": expected_count,
            "last_generation_id": expected_last_generation_id,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"RNG checkpoint mismatch for {key}: {path}")
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        self.torch.set_rng_state(state["torch_cpu_rng_state"])
        self.torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])


def validate_resume_prefix(
    assigned: list[Mapping[str, object]],
    existing_records: list[Mapping[str, object]],
) -> None:
    """Require a worker output to be the exact prefix of its assigned shard."""

    if len(existing_records) > len(assigned):
        raise ValueError("Worker output contains more records than its shard")
    expected_ids = [str(task["generation_id"]) for task in assigned[: len(existing_records)]]
    actual_ids = [str(record["generation_id"]) for record in existing_records]
    if actual_ids != expected_ids:
        raise ValueError("Worker output is not an ordered prefix of its assigned shard")


def worker(args: argparse.Namespace) -> None:
    if args.worker_id < 0 or args.worker_id >= args.world_size:
        raise ValueError("worker_id must be in [0, world_size)")
    manifest = list(read_jsonl(Path(args.manifest)))
    assigned = [
        task
        for task in manifest
        if task["world_size"] == args.world_size and task["shard_id"] == args.worker_id
    ]
    if not assigned:
        raise ValueError("No tasks assigned to worker")
    if any(task["model_key"] != args.model_key for task in assigned):
        raise ValueError("Manifest model_key does not match worker model_key")
    output = Path(args.output)
    existing_records = list(read_jsonl(output)) if output.exists() else []
    validate_resume_prefix(assigned, existing_records)
    recorded = recorded_generation_ids(output)
    todo = list(pending_tasks(assigned, recorded))
    if not todo:
        print(f"Worker {args.worker_id}: all {len(assigned)} tasks already recorded")
        return
    generator = LocalHFGenerator(
        args.model_key,
        seed_mode=args.seed_mode,
        worker_seed=args.worker_seed,
    )
    rng_state_path = (
        Path(args.rng_state)
        if args.rng_state
        else output.with_name(output.name + ".rng.pt")
    )
    if args.seed_mode == "seed_once" and existing_records:
        generator.restore_rng_checkpoint(
            rng_state_path,
            expected_count=len(existing_records),
            expected_last_generation_id=str(existing_records[-1]["generation_id"]),
        )
    for offset, task in enumerate(todo, start=1):
        try:
            response = generator.generate(str(task["prompt"]), int(task["seed"]))
            record = result_from_response(task, response)
            record.update(
                {
                    "seed_mode": args.seed_mode,
                    "worker_seed": args.worker_seed,
                    "worker_id": args.worker_id,
                }
            )
        except Exception as exc:
            record = result_from_failure(task, exc)
            record.update(
                {
                    "seed_mode": args.seed_mode,
                    "worker_seed": args.worker_seed,
                    "worker_id": args.worker_id,
                }
            )
            append_jsonl_record(output, record)
            raise
        append_jsonl_record(output, record)
        if args.seed_mode == "seed_once":
            generator.save_rng_checkpoint(
                rng_state_path,
                completed_count=len(existing_records) + offset,
                last_generation_id=str(task["generation_id"]),
            )
    print(f"Worker {args.worker_id}: wrote {len(todo)} results to {output}")


def merge(args: argparse.Namespace) -> None:
    manifest = list(read_jsonl(Path(args.manifest)))
    records = []
    for path in args.worker_output:
        records.extend(read_jsonl(Path(path)))
    merged, audit = merge_results(manifest, records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_jsonl = output_dir / "responses.jsonl"
    merged_csv = output_dir / "responses.csv"
    summary_path = output_dir / "summary.json"
    write_jsonl_exclusive(merged_jsonl, merged)
    write_csv_exclusive(merged_csv, merged)
    summary = summarize_results(manifest, merged)
    summary["merge_audit"] = audit
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--model-key", choices=MODEL_SPECS, default="official")
    prepare_parser.add_argument("--statement-limit", type=int)
    prepare_parser.add_argument("--world-size", type=int, default=2)
    prepare_parser.add_argument(
        "--newline-mode",
        choices=["literal", "newline", "escaped-to-newline"],
        default="literal",
        help=(
            "Reconstructed prompt formatting: literal preserves template escapes "
            "(default); newline converts template backslash-n before substitution. "
            "escaped-to-newline is an alias for newline. Original behavior is unresolved."
        ),
    )
    prepare_parser.set_defaults(func=prepare)

    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--manifest", required=True)
    worker_parser.add_argument("--model-key", choices=MODEL_SPECS, required=True)
    worker_parser.add_argument("--worker-id", type=int, required=True)
    worker_parser.add_argument("--world-size", type=int, default=2)
    worker_parser.add_argument(
        "--seed-mode",
        choices=SEED_MODES,
        default="reset_each_generation",
        help="Default preserves the authoritative per-generation seed reset.",
    )
    worker_parser.add_argument("--worker-seed", type=int, default=AUTHORITATIVE_SEED)
    worker_parser.add_argument(
        "--rng-state",
        help="Optional seed_once RNG checkpoint path; defaults beside worker output.",
    )
    worker_parser.add_argument("--output", required=True)
    worker_parser.set_defaults(func=worker)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--manifest", required=True)
    merge_parser.add_argument("--worker-output", action="append", required=True)
    merge_parser.add_argument("--output-dir", required=True)
    merge_parser.set_defaults(func=merge)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
