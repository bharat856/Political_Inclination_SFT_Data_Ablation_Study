"""Resume one prepared ProbVAA model on one visible GPU.

This is execution orchestration only.  It invokes the existing authoritative
worker for manifest shard 0 and then shard 1, preserving the prepared
two-shard manifests and append-only worker outputs.  After both shards finish,
it invokes the existing merge command.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "code" / "evaluation" / "run_probvaa_authoritative.py"
PYTHON = ROOT / "envs" / "classifier" / "bin" / "python"
EXPECTED_TASKS = 172_080
EXPECTED_SHARD_TASKS = 86_040


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_manifest_and_outputs(model_dir: Path, model_key: str) -> dict:
    manifest = read_jsonl(model_dir / "tasks.jsonl")
    if len(manifest) != EXPECTED_TASKS:
        raise RuntimeError(f"Expected {EXPECTED_TASKS} manifest tasks")
    manifest_ids = [str(task["generation_id"]) for task in manifest]
    if len(set(manifest_ids)) != EXPECTED_TASKS:
        raise RuntimeError("Manifest task IDs are not unique")
    shard_counts = {}
    completed_counts = {}
    for shard_id in (0, 1):
        assigned = [
            task for task in manifest
            if task["world_size"] == 2 and task["shard_id"] == shard_id
        ]
        if len(assigned) != EXPECTED_SHARD_TASKS:
            raise RuntimeError(f"Shard {shard_id} has {len(assigned)} tasks")
        if any(task["model_key"] != model_key for task in assigned):
            raise RuntimeError(f"Shard {shard_id} model key mismatch")
        if any(
            task["newline_mode"] != "literal"
            or task["seed"] != 42
            or task["prompt_type"] != "personal"
            for task in assigned
        ):
            raise RuntimeError(f"Shard {shard_id} methodology mismatch")
        records = read_jsonl(model_dir / f"worker-{shard_id}.jsonl")
        expected_prefix = [str(task["generation_id"]) for task in assigned[:len(records)]]
        actual = [str(record["generation_id"]) for record in records]
        if actual != expected_prefix:
            raise RuntimeError(f"Shard {shard_id} output is not an exact resume prefix")
        if len(actual) != len(set(actual)):
            raise RuntimeError(f"Shard {shard_id} output has duplicate task IDs")
        shard_counts[str(shard_id)] = len(assigned)
        completed_counts[str(shard_id)] = len(records)
    return {
        "manifest_tasks": len(manifest),
        "shard_tasks": shard_counts,
        "completed_before_launch": completed_counts,
    }


def worker_command(model_dir: Path, model_key: str, shard_id: int) -> list[str]:
    return [
        str(PYTHON), "-u", str(RUNNER), "worker",
        "--manifest", str(model_dir / "tasks.jsonl"),
        "--model-key", model_key,
        "--worker-id", str(shard_id),
        "--world-size", "2",
        "--seed-mode", "reset_each_generation",
        "--worker-seed", "42",
        "--output", str(model_dir / f"worker-{shard_id}.jsonl"),
    ]


def run_child(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] launch: {' '.join(command)}\n")
        log.flush()
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            code = child.wait()
        except KeyboardInterrupt:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.terminate()
            raise
    if code != 0:
        raise RuntimeError(f"Child exited with status {code}: {' '.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--model-key",
        required=True,
        choices=("official", "Full", "NoPol", "NoLeft", "NoRight"),
    )
    parser.add_argument("--physical-gpu", required=True, type=int)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={args.physical_gpu}, found {visible!r}"
        )
    model_dir = Path(args.model_dir).resolve()
    audit = validate_manifest_and_outputs(model_dir, args.model_key)
    state_path = model_dir / "single_gpu_execution_state.json"
    state = {
        "model_key": args.model_key,
        "physical_gpu": args.physical_gpu,
        "methodology": {
            "seed_mode": "reset_each_generation",
            "seed": 42,
            "newline_mode": "literal",
            "world_size_in_preserved_manifest": 2,
            "execution": "two manifest shards sequentially on one physical GPU",
        },
        "prelaunch_audit": audit,
        "status": "running",
        "started_at": now(),
    }
    atomic_json(state_path, state)
    try:
        for shard_id in (0, 1):
            state["active_shard"] = shard_id
            atomic_json(state_path, state)
            run_child(
                worker_command(model_dir, args.model_key, shard_id),
                model_dir / f"worker-{shard_id}.log",
            )
        merged = model_dir / "merged"
        summary = merged / "summary.json"
        if not summary.is_file():
            if merged.exists():
                raise RuntimeError(f"Refusing to overwrite incomplete merge: {merged}")
            merge_command = [
                str(PYTHON), str(RUNNER), "merge",
                "--manifest", str(model_dir / "tasks.jsonl"),
                "--worker-output", str(model_dir / "worker-0.jsonl"),
                "--worker-output", str(model_dir / "worker-1.jsonl"),
                "--output-dir", str(merged),
            ]
            run_child(merge_command, model_dir / "merge.log")
        state.pop("active_shard", None)
        state["status"] = "complete"
        state["completed_at"] = now()
        atomic_json(state_path, state)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["interrupted_at"] = now()
        atomic_json(state_path, state)
        raise
    except BaseException as exc:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
