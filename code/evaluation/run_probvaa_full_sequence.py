"""Fail-fast sequential launcher for the approved full ProbVAA evaluation.

This wrapper does not implement evaluation methodology. It invokes the tested
prepare/worker/merge implementation in ``run_probvaa_authoritative.py`` and
validates every model before advancing to the next one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "code" / "evaluation" / "run_probvaa_authoritative.py"
FINAL_REPORT = ROOT / "code" / "evaluation" / "PROBVAA_FULL_EXECUTION_REPORT.md"
AUTHORITATIVE_COMMIT = "b44b81f0169228a8565c0c810bf5499b38e3885f"
EXPECTED_TASKS = 172_080
EXPECTED_SHARD_TASKS = 86_040
MODEL_ORDER = (
    ("Official", "official"),
    ("Full", "Full"),
    ("NoPol", "NoPol"),
    ("NoLeft", "NoLeft"),
    ("NoRight", "NoRight"),
)
MODEL_IDENTITIES = {
    "official": {
        "model": "allenai/Olmo-3-7B-Instruct-SFT",
        "model_path": "validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT",
        "base_model": None,
        "adapter": None,
        "prompt_mode": "chat_template",
    },
    "Full": {
        "model": "Full",
        "model_path": "hf_cache/allenai_Olmo-3-1025-7B",
        "base_model": "allenai/Olmo-3-1025-7B",
        "adapter": "outputs/sft_runs/olmo3_instruct_full_4gpu_v4",
        "prompt_mode": "raw",
    },
    "NoPol": {
        "model": "NoPol",
        "model_path": "hf_cache/allenai_Olmo-3-1025-7B",
        "base_model": "allenai/Olmo-3-1025-7B",
        "adapter": "outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5",
        "prompt_mode": "raw",
    },
    "NoLeft": {
        "model": "NoLeft",
        "model_path": "hf_cache/allenai_Olmo-3-1025-7B",
        "base_model": "allenai/Olmo-3-1025-7B",
        "adapter": "outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4",
        "prompt_mode": "raw",
    },
    "NoRight": {
        "model": "NoRight",
        "model_path": "hf_cache/allenai_Olmo-3-1025-7B",
        "base_model": "allenai/Olmo-3-1025-7B",
        "adapter": "outputs/sft_runs/olmo3_instruct_noright_4gpu_v5",
        "prompt_mode": "raw",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_manifest(path: Path, model_key: str) -> dict[str, object]:
    task_ids: set[str] = set()
    statement_ids: set[str] = set()
    formulation_ids: set[str] = set()
    template_ids: set[str] = set()
    shard_counts = {0: 0, 1: 0}
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            task = json.loads(line)
            rows += 1
            generation_id = str(task["generation_id"])
            if generation_id in task_ids:
                raise RuntimeError(f"Duplicate manifest task ID: {generation_id}")
            task_ids.add(generation_id)
            if task["model_key"] != model_key:
                raise RuntimeError(f"Manifest model mismatch in {path}")
            if task["newline_mode"] != "literal" or task["seed"] != 42:
                raise RuntimeError(f"Manifest methodology mismatch in {path}")
            if task["prompt_type"] != "personal" or task["world_size"] != 2:
                raise RuntimeError(f"Manifest selection/sharding mismatch in {path}")
            shard_counts[int(task["shard_id"])] += 1
            statement_ids.add(str(task["statement_id"]))
            formulation_ids.add(str(task["formulation_id"]))
            template_ids.add(str(task["template_id"]))
    if rows != EXPECTED_TASKS or len(task_ids) != EXPECTED_TASKS:
        raise RuntimeError(f"Expected {EXPECTED_TASKS} unique tasks in {path}")
    if shard_counts != {0: EXPECTED_SHARD_TASKS, 1: EXPECTED_SHARD_TASKS}:
        raise RuntimeError(f"Unexpected shard counts in {path}: {shard_counts}")
    if len(statement_ids) != 239 or len(formulation_ids) != 956:
        raise RuntimeError(f"Unexpected statement/formulation IDs in {path}")
    if template_ids != {"1", "2", "3", "7", "8", "9"}:
        raise RuntimeError(f"Unexpected template IDs in {path}: {template_ids}")
    return {
        "rows": rows,
        "unique_task_ids": len(task_ids),
        "statement_ids": len(statement_ids),
        "formulation_ids": len(formulation_ids),
        "template_ids": sorted(template_ids),
        "shard_counts": shard_counts,
        "sha256": sha256(path),
    }


def validate_summary(path: Path) -> dict:
    summary = load_json(path)
    audit = summary["merge_audit"]
    expected = {
        "expected": EXPECTED_TASKS,
        "recorded": EXPECTED_TASKS,
        "completed": EXPECTED_TASKS,
        "failed": 0,
        "missing": 0,
        "duplicate_ids": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Failed merge audit in {path}: {audit}")
    if (
        summary["agree_count"]
        + summary["disagree_count"]
        + summary["na_count"]
        != EXPECTED_TASKS
    ):
        raise RuntimeError(f"Non-exhaustive classifications in {path}")
    return summary


def worker_command(
    python: Path,
    manifest: Path,
    model_key: str,
    worker_id: int,
    output: Path,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(RUNNER),
        "worker",
        "--manifest",
        str(manifest),
        "--model-key",
        model_key,
        "--worker-id",
        str(worker_id),
        "--world-size",
        "2",
        "--seed-mode",
        "reset_each_generation",
        "--worker-seed",
        "42",
        "--output",
        str(output),
    ]


def terminate_workers(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 30
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()


def run_workers(commands: list[list[str]], logs: list[Path]) -> None:
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        for worker_id, (command, log_path) in enumerate(zip(commands, logs)):
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            handle.write(f"\n[{now()}] launch: {' '.join(command)}\n")
            handle.flush()
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(worker_id),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
            )
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process.wait): process for process in processes}
            pending = set(futures)
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    code = future.result()
                    if code != 0:
                        terminate_workers([futures[item] for item in pending])
                        raise RuntimeError(f"Worker exited with status {code}")
    except BaseException:
        terminate_workers(processes)
        raise
    finally:
        for handle in handles:
            handle.close()


def merge_model(python: Path, model_dir: Path) -> dict:
    merged = model_dir / "merged"
    summary_path = merged / "summary.json"
    if summary_path.is_file():
        return validate_summary(summary_path)
    if merged.exists():
        raise RuntimeError(f"Incomplete merge directory requires audit: {merged}")
    command = [
        str(python),
        str(RUNNER),
        "merge",
        "--manifest",
        str(model_dir / "tasks.jsonl"),
        "--worker-output",
        str(model_dir / "worker-0.jsonl"),
        "--worker-output",
        str(model_dir / "worker-1.jsonl"),
        "--output-dir",
        str(merged),
    ]
    with (model_dir / "merge.log").open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Merge failed with status {completed.returncode}")
    return validate_summary(summary_path)


def write_final_report(output_root: Path, state: dict) -> None:
    lines = [
        "# Full Authoritative ProbVAA Execution Report",
        "",
        f"Completed: {now()}  ",
        f"Experiment: `{output_root.name}`  ",
        f"Authoritative commit: `{AUTHORITATIVE_COMMIT}`",
        "",
        "## Methodological configuration",
        "",
        "- 239 statements, four available formulations, six personal templates, and 30 generations per prompt.",
        "- 172,080 tasks per model; two deterministic modulo-two shards of 86,040 tasks.",
        "- Effective generation: `do_sample=True`, `temperature=1.0`, `top_p=None`, `max_new_tokens=10`.",
        "- Seed 42 reset immediately before every generation (`reset_each_generation`).",
        "- Literal backslash-`n` sequences preserved (`newline_mode=literal`).",
        "- Official model uses its chat template; base-derived LoRA models use raw prompts.",
        "- Only the unchanged authoritative `map_response` classifier is used; no worldview aggregation or ideology scoring.",
        "- This is a restricted adaptation, not an exact reproduction of all historical evaluation settings.",
        "",
        "## Model results",
        "",
        "| Model | Manifest SHA-256 | Runtime | Agree | Disagree | NA | Missing | Failed | Identical prompts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for display, key in MODEL_ORDER:
        item = state["models"][display]
        summary = item["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    display,
                    f"`{item['manifest']['sha256']}`",
                    f"{item['runtime_seconds'] / 3600:.3f} h",
                    f"{summary['agree_count']:,} ({summary['agree_percentage']:.4f}%)",
                    f"{summary['disagree_count']:,} ({summary['disagree_percentage']:.4f}%)",
                    f"{summary['na_count']:,} ({summary['na_percentage']:.4f}%)",
                    str(summary["missing_generations"]),
                    str(summary["failed_generations"]),
                    f"{summary['prompts_identical_across_all_iterations']:,}/{summary['total_prompt_groups']:,} ({summary['percentage_complete_prompts_identical_across_all_iterations']:.4f}%)",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Exact model identities", ""])
    for display, key in MODEL_ORDER:
        identity = MODEL_IDENTITIES[key]
        lines.extend(
            [
                f"### {display}",
                "",
                f"- Model: `{identity['model']}`",
                f"- Model/base path: `{identity['model_path']}`",
                f"- Adapter: `{identity['adapter']}`",
                f"- Prompt mode: `{identity['prompt_mode']}`",
                f"- Output: `{output_root / display}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Resume and control",
            "",
            f"- Progress: `wc -l {output_root}/*/worker-*.jsonl`",
            f"- Stop safely: `tmux send-keys -t {state['session_name']} C-c`",
            f"- Resume: rerun the exact launch command recorded in `{output_root / 'launch_command.txt'}`.",
            "",
            "Detailed NA breakdowns by template, statement, and formulation, plus per-prompt raw-response diversity, are stored in each model's `merged/summary.json`.",
        ]
    )
    if FINAL_REPORT.exists():
        return
    with FINAL_REPORT.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--session-name", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    python = ROOT / "envs" / "classifier" / "bin" / "python"
    state_path = output_root / "execution_state.json"
    state = load_json(state_path) if state_path.is_file() else {
        "experiment": output_root.name,
        "session_name": args.session_name,
        "authoritative_commit": AUTHORITATIVE_COMMIT,
        "started_at": now(),
        "models": {},
    }
    try:
        for display, model_key in MODEL_ORDER:
            model_dir = output_root / display
            manifest_info = validate_manifest(model_dir / "tasks.jsonl", model_key)
            summary_path = model_dir / "merged" / "summary.json"
            if summary_path.is_file():
                summary = validate_summary(summary_path)
                item = state["models"].setdefault(display, {})
                item.update(
                    {
                        "status": "complete",
                        "manifest": manifest_info,
                        "summary": summary,
                        "resumed_validation_at": now(),
                        "runtime_seconds": item.get("runtime_seconds", 0.0),
                    }
                )
                atomic_json(state_path, state)
                continue
            commands = [
                worker_command(
                    python,
                    model_dir / "tasks.jsonl",
                    model_key,
                    worker_id,
                    model_dir / f"worker-{worker_id}.jsonl",
                )
                for worker_id in (0, 1)
            ]
            atomic_json(
                model_dir / "commands.json",
                {
                    "worker_0": commands[0],
                    "worker_1": commands[1],
                    "seed_mode": "reset_each_generation",
                    "seed": 42,
                    "gpu_ids": [0, 1],
                },
            )
            item = state["models"].setdefault(display, {})
            item.update(
                {
                    "status": "running",
                    "started_at": now(),
                    "manifest": manifest_info,
                    "identity": MODEL_IDENTITIES[model_key],
                }
            )
            atomic_json(state_path, state)
            start = time.monotonic()
            run_workers(
                commands,
                [model_dir / "worker-0.log", model_dir / "worker-1.log"],
            )
            summary = merge_model(python, model_dir)
            item.update(
                {
                    "status": "complete",
                    "completed_at": now(),
                    "runtime_seconds": time.monotonic() - start,
                    "summary": summary,
                }
            )
            atomic_json(state_path, state)
        state["status"] = "complete"
        state["completed_at"] = now()
        atomic_json(state_path, state)
        write_final_report(output_root, state)
        print(f"[{now()}] all five models completed successfully", flush=True)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["interrupted_at"] = now()
        atomic_json(state_path, state)
        print(f"[{now()}] interrupted safely; completed records preserved", flush=True)
        raise
    except BaseException as exc:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(state_path, state)
        print(f"[{now()}] FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise


if __name__ == "__main__":
    main()
