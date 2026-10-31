# ProbVAA Five-GPU Migration Status

Status snapshot: `2026-10-19T23:56:15+02:00`  
Experiment directory: `outputs/probvaa_full_reset42_literal_20261019/`

## Migration result

The sequential ProbVAA execution was safely migrated to five concurrent,
single-GPU model evaluations. The former tmux session
`probvaa_full_reset42_literal_20261019` was interrupted using `Ctrl-C` and no
longer exists.

Before the new sessions were launched, GPUs 0--4 were verified to have 48,592
MiB free each, zero utilization, and no conflicting compute processes. After
launch, all five models loaded successfully and were actively generating. Each
model used approximately 14.3--14.5 GiB of GPU memory. No new runtime errors
were detected during post-launch verification.

## GPU and session allocation

| GPU | Model | tmux session | Target tasks | Completed records at snapshot |
|---:|---|---|---:|---:|
| 0 | Official OLMo-3-7B-Instruct-SFT | `probvaa5_official_gpu0` | 172,080 | 28,114 |
| 1 | Full | `probvaa5_full_gpu1` | 172,080 | 3,790 |
| 2 | NoPol | `probvaa5_nopol_gpu2` | 172,080 | 4,013 |
| 3 | NoLeft | `probvaa5_noleft_gpu3` | 172,080 | 3,448 |
| 4 | NoRight | `probvaa5_noright_gpu4` | 172,080 | 3,509 |

These record counts are a timestamped snapshot. The output files continue to
grow while the evaluations run.

At the snapshot time, all five tmux sessions remained present and GPUs 0--4
showed active utilization.

## Preservation and resume verification

The interrupted Official evaluation contained:

- 3,796 completed records in manifest shard 0.
- 3,842 completed records in manifest shard 1.
- 7,638 total records preserved at migration time.

Both existing Official output files were valid JSONL, contained unique task
IDs, and were exact ordered prefixes of their corresponding 86,040-task
manifest shards. They contained no recorded generation failures. The new
Official process resumed shard 0 by appending only pending task IDs. When shard
0 finishes, it will resume shard 1 after its preserved 3,842-record prefix.

Every model manifest was independently verified to contain:

- 172,080 rows and 172,080 unique generation IDs.
- Two fixed manifest shards containing 86,040 tasks each.
- 239 underlying statement IDs.
- 956 formulation IDs.
- Personal template IDs `1`, `2`, `3`, `7`, `8`, and `9` only.

No manifest was regenerated or overwritten.

## Preserved methodology

The migration changed only execution orchestration. The existing evaluation
worker continues to use:

- `seed_mode=reset_each_generation`.
- Seed 42 immediately before every generation.
- `newline_mode=literal` from the prepared manifests.
- Six personal prompt templates.
- Four formulations per political statement.
- Thirty generations per prompt.
- `do_sample=True`, temperature 1.0, `top_p=None`, and
  `max_new_tokens=10` as the existing effective generation configuration.
- The unchanged authoritative response classifier.
- Chat-template prompting for the Official Instruct-SFT checkpoint.
- Raw prompting with the verified pretrained base for the four LoRA adapters.

The execution-only wrapper is
`code/evaluation/run_probvaa_model_single_gpu.py`. It validates each existing
manifest and output prefix, runs manifest shard 0 and shard 1 sequentially on
the assigned physical GPU, and then invokes the existing merge command. It
does not implement or alter prompt formatting, generation, or classification.

## Output and log locations

Each model has independent append-only outputs, shard logs, execution state,
and merge output under its model directory:

| Model | Directory |
|---|---|
| Official | `outputs/probvaa_full_reset42_literal_20261019/Official/` |
| Full | `outputs/probvaa_full_reset42_literal_20261019/Full/` |
| NoPol | `outputs/probvaa_full_reset42_literal_20261019/NoPol/` |
| NoLeft | `outputs/probvaa_full_reset42_literal_20261019/NoLeft/` |
| NoRight | `outputs/probvaa_full_reset42_literal_20261019/NoRight/` |

Within each directory:

- `worker-0.jsonl` and `worker-1.jsonl` contain append-only raw results.
- `worker-0.log` and `worker-1.log` contain worker output.
- `model-orchestrator.log` contains model-controller output.
- `single_gpu_execution_state.json` records the GPU, methodology, current
  shard, and run state.
- `merged/` will be created by the unchanged merger after both shards finish.

The Official `worker-0.log` retains the expected historical
`KeyboardInterrupt` from the safe migration. The subsequent launch entry marks
the current run; no new error followed that launch during verification.

## Monitoring commands

Count completed records:

```bash
wc -l outputs/probvaa_full_reset42_literal_20261019/*/worker-*.jsonl
```

Inspect GPU activity:

```bash
watch -n 5 nvidia-smi
```

List the five evaluation sessions:

```bash
tmux list-sessions | grep probvaa5_
```

Follow a worker log, replacing `Official` with another model when required:

```bash
tail -f outputs/probvaa_full_reset42_literal_20261019/Official/worker-0.log
```

Inspect per-model orchestration state:

```bash
cat outputs/probvaa_full_reset42_literal_20261019/Official/single_gpu_execution_state.json
```

## Safe stopping commands

```bash
tmux send-keys -t probvaa5_official_gpu0 C-c
tmux send-keys -t probvaa5_full_gpu1 C-c
tmux send-keys -t probvaa5_nopol_gpu2 C-c
tmux send-keys -t probvaa5_noleft_gpu3 C-c
tmux send-keys -t probvaa5_noright_gpu4 C-c
```

The wrapper signals the active worker and preserves all completed JSONL
records. Partial result files must not be deleted before resuming.

## Exact resume commands

Run a command only after confirming that its target GPU is free and its tmux
session name does not already exist.

### Official on GPU 0

```bash
tmux new-session -d -s probvaa5_official_gpu0 -c /fs/scratch/users/bharatpi/causal_political_content "exec env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_model_single_gpu.py --model-dir outputs/probvaa_full_reset42_literal_20261019/Official --model-key official --physical-gpu 0 >> outputs/probvaa_full_reset42_literal_20261019/Official/model-orchestrator.log 2>&1"
```

### Full on GPU 1

```bash
tmux new-session -d -s probvaa5_full_gpu1 -c /fs/scratch/users/bharatpi/causal_political_content "exec env CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_model_single_gpu.py --model-dir outputs/probvaa_full_reset42_literal_20261019/Full --model-key Full --physical-gpu 1 >> outputs/probvaa_full_reset42_literal_20261019/Full/model-orchestrator.log 2>&1"
```

### NoPol on GPU 2

```bash
tmux new-session -d -s probvaa5_nopol_gpu2 -c /fs/scratch/users/bharatpi/causal_political_content "exec env CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_model_single_gpu.py --model-dir outputs/probvaa_full_reset42_literal_20261019/NoPol --model-key NoPol --physical-gpu 2 >> outputs/probvaa_full_reset42_literal_20261019/NoPol/model-orchestrator.log 2>&1"
```

### NoLeft on GPU 3

```bash
tmux new-session -d -s probvaa5_noleft_gpu3 -c /fs/scratch/users/bharatpi/causal_political_content "exec env CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_model_single_gpu.py --model-dir outputs/probvaa_full_reset42_literal_20261019/NoLeft --model-key NoLeft --physical-gpu 3 >> outputs/probvaa_full_reset42_literal_20261019/NoLeft/model-orchestrator.log 2>&1"
```

### NoRight on GPU 4

```bash
tmux new-session -d -s probvaa5_noright_gpu4 -c /fs/scratch/users/bharatpi/causal_political_content "exec env CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_model_single_gpu.py --model-dir outputs/probvaa_full_reset42_literal_20261019/NoRight --model-key NoRight --physical-gpu 4 >> outputs/probvaa_full_reset42_literal_20261019/NoRight/model-orchestrator.log 2>&1"
```

## Files changed for migration

- Added `code/evaluation/run_probvaa_model_single_gpu.py` for execution
  orchestration.
- Added this status report.

The authoritative repository, prepared manifests, scientific evaluation
pipeline, prompts, classifier, model weights, and adapters were not modified.
