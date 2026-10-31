# Authoritative ProbVAA Evaluation Implementation Report

Date of implementation audit: 2026-10-19  
Status: CPU implementation and one-prompt GPU smoke test validated. The
subsequently approved 5,040-response pilot completed successfully; see
`PROBVAA_AUTHORITATIVE_PILOT_RESULTS.md`.

## 1. Authoritative source

The sole methodological reference is the fresh clone at:

`references/political_content_training_data_fresh/`

- Remote: `https://github.com/tceron/political_content_training_data.git`
- Branch: `main`
- Pinned and checked-out commit: `b44b81f0169228a8565c0c810bf5499b38e3885f`
- Working tree at verification: clean (`git status --short` returned no paths)

No code, data, templates, or assumptions were taken from
`references/eval_political_worldviews/` or the legacy evaluator.

### Authoritative files used

| Purpose | Source path | Evidence |
|---|---|---|
| Statements and four formulations | `references/political_content_training_data_fresh/model_behavior/data/prob_vaa_en.csv` | 956 rows, 239 `unique_id` groups, exactly one original and three paraphrases per group |
| Prompt templates | `references/political_content_training_data_fresh/model_behavior/data/en_clf_templates_final.csv` | `prompt_type` is the type column; exact filter `row["prompt_type"] == "personal"` leaves six rows in CSV order |
| Evaluation loop and requested decoding configuration | `references/political_content_training_data_fresh/model_behavior/classification_with_model_prompting.py:67-153` | `main`; configuration at lines 77-82, 30-call loop at lines 125-132, mapping at line 141 |
| Response mapping | `references/political_content_training_data_fresh/model_behavior/classification_with_model_prompting.py:13-58` | `prompt_formats` and `map_response` |
| HF prompt preparation, seed, generation, and decoding | `references/political_content_training_data_fresh/model_behavior/llm_utils.py:263-340` | `_set_pad_token_id`, `update_seed`, and `_HFModel.__call__` |

The upstream evaluator imports `format_prompt` at
`classification_with_model_prompting.py:2` and calls it at lines 122-123, but
the function is absent from the pinned repository and all locally reachable
history. The local implementation is therefore explicitly isolated and
documented below.

## 2. Files implemented

| File | Purpose |
|---|---|
| `code/evaluation/probvaa_authoritative_selection.py` | Exact `prompt_type == "personal"` selection, preserving source rows and order |
| `code/evaluation/probvaa_authoritative_pipeline.py` | CPU-safe data loading, prompt formatting, exact isolated `map_response`, task IDs/sharding, resume/merge validation, and NA reporting |
| `code/evaluation/run_probvaa_authoritative.py` | Offline standalone/PEFT model loading and prepare/worker/merge CLI |
| `code/evaluation/test_probvaa_authoritative_selection.py` | Template-selection tests |
| `code/evaluation/test_probvaa_authoritative_pipeline.py` | Protocol, parser-parity, sharding, resume, merge, and reporting tests |
| `code/evaluation/PROBVAA_IMPLEMENTATION_REPORT.md` | This report |

The authoritative fresh clone was not modified.

## 3. Exact evaluation protocol

| Quantity | Verified value |
|---|---:|
| Underlying political statements | 239 |
| Formulations per statement | 4 (one original plus three paraphrases) |
| Statement/formulation rows | 956 |
| Personal templates | 6 |
| Unique prompts | 5,736 |
| Iterations per prompt | 30 |
| Responses per model | 172,080 |

The implementation reads source CSV order directly. It filters templates only
by `prompt_type == "personal"`; no template ID or positional selection is used.
The retained source template IDs, in their existing CSV order, are 1, 2, 3, 7,
8, and 9. This list is an observed result, not a selection rule. All six
impersonal rows are excluded.

Each generation ID is deterministic and reversible:

`model/condition :: formulation ID :: template ID :: sample iteration`

The full manifest test verified 172,080 distinct generation IDs and 5,736
distinct formulation-template prompt pairs.

## 4. Prompt formatting

Every authoritative template has exactly one `{statement}` placeholder. The
implemented default performs exactly one direct substitution and does not
otherwise normalize, trim, or rewrite either string.

The CSV contains literal backslash-plus-`n` sequences. No evidence in the
pinned repository resolves whether upstream `format_prompt` converted these to
line-feed characters. The methodological default is therefore
`--newline-mode literal`, which preserves the CSV text byte-for-byte except for
the placeholder substitution. An explicit `escaped-to-newline` mode exists for
future sensitivity testing, but it is not used by the prepared pilot.

This missing upstream function and unresolved newline intent remain a
reproduction limitation.

## 5. Generation behavior

The source script declares:

```python
{
    "do_sample": False,
    "max_new_tokens": 10,
    "temperature": 1.0,
    "top_p": None,
}
```

This is not the effective configuration. In
`llm_utils.py:315-324`, every nonzero temperature forces
`generation_config["do_sample"] = True`. The effective configuration is:

```python
{
    "do_sample": True,
    "max_new_tokens": 10,
    "temperature": 1.0,
    "top_p": None,
}
```

The runner reproduces this mutation rather than silently honoring the initial
`False`. It also passes one return sequence, the tokenizer EOS ID, and the
upstream pad-token fallback.

The source calls the model 30 times and resets seed 42 immediately before every
call (`classification_with_model_prompting.py:125-132` and
`llm_utils.py:274-279`). Thus the protocol uses sampling, but repeated calls for
the same prompt begin from the same RNG state. They are expected to produce the
same response under deterministic hardware/software execution. This is a
finding, not a protocol change: all 30 tasks are retained.

## 6. Response classification and NA reporting

`map_response` was copied without behavioral changes from the pinned source.
It lowercases generated text, removes listed prompt phrases, tokenizes with
`[a-z]+`, returns the first recognized positive or negative keyword, and applies
its limited immediately preceding `not`/`don't` handling. It returns:

- `1`: Agree
- `0`: Disagree
- `None`: NA

There is no secondary classifier model. Mapping is performed directly on raw
generated text. The regression test extracts the authoritative function by AST
without importing its unrelated dependencies, then compares both functions on:
Agree, Disagree, Support, Oppose, Beneficial, Harmful, Favorable, Detrimental,
both tested negations, both keyword orders, unknown/empty text, mixed case,
quotation marks, and punctuation. All comparisons passed.

Only completed generations enter the Agree/Disagree/NA denominator. NA means
exactly `map_response(response) is None`. Failures and missing task IDs are
reported separately. The summary contains total counts and NA rates grouped by
template, underlying statement, and formulation. Raw `response` and
`response_mapped` are preserved, with the upstream-compatible leading columns:
`prompt_id`, `sample_id`, `response_mapped`, `response`, and
`prompt_description_id`.

## 7. Model support and prompt compatibility

### Official standalone checkpoint

- Model ID: `allenai/Olmo-3-7B-Instruct-SFT`
- Local path: `validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT`
- Loading: standalone `AutoModelForCausalLM`; no PEFT adapter
- Prompt handling: one user message through the model's local chat template,
  with `add_generation_prompt=True`, matching the authoritative name-based SFT
  branch at `llm_utils.py:289-293`

### Six experimental adapters

The runner supports Full, NoPol, NoIdeology, NoLeft, NoRight, and LRBalanced.
Each loads the local `hf_cache/allenai_Olmo-3-1025-7B` base, then attaches the
condition's final PEFT adapter read-only with `is_trainable=False`.

The local official tokenizer has a chat template. The local pretrained base
tokenizer and adapter tokenizers do not. The authoritative implementation uses
raw tokenization when its model-name argument contains none of `sft`, `dpo`, or
`instruct`; therefore adapter evaluation uses raw prompt tokenization and does
not invent a generic chat template. This is a model compatibility adaptation
derived from the authoritative branching behavior and the verified local
tokenizer metadata.

All model operations are offline (`local_files_only=True`); BF16 is requested,
no quantization is introduced, and every worker loads its own model instance.
Static offline loading succeeded for both configurations/tokenizers and all six
PEFT configs. Model weights were not loaded during this preparation.

## 8. Two-GPU execution design

Two independent processes are used, not tensor parallelism. Task index modulo
two determines the shard. Each process must see exactly one GPU and loads an
independent model instance. Worker files are separate. A durable JSONL record is
appended after each completed or failed attempt. Existing generation IDs are
not regenerated on resume. Merge rejects unknown IDs, metadata mismatches, and
duplicates across worker files; it then restores manifest order and reports
missing tasks.

The model checkpoint occupies approximately 14 GB on disk (13,596,307,712
bytes across three BF16 safetensor shards). The one-prompt smoke test measured
14,664,257,536 bytes peak allocated GPU memory. Each RTX A6000 exposes
50,952,536,064 bytes, so the verified configuration fits on one GPU with ample
headroom.

## 9. CPU validation results

Command:

```bash
cd code/evaluation
/fs/scratch/users/bharatpi/causal_political_content/envs/classifier/bin/python -m unittest test_probvaa_authoritative_selection.py test_probvaa_authoritative_pipeline.py -v
```

Result: **15 tests passed**, 0 failures, in 0.636 seconds. Python syntax
compilation also passed.

Verified invariants include:

1. 956 rows, 239 statements, four formulations per statement.
2. Six personal templates and no impersonal templates.
3. 5,736 unique prompts and 172,080 unique full-run tasks.
4. Placeholder substitution, text preservation, and deterministic formatting.
5. Exact parser parity for all required regression inputs.
6. Disjoint, complete, evenly sized two-worker shards.
7. Duplicate detection, safe merge, and nonduplicating resume behavior.
8. Failed/missing generations remain separate from NA.
9. Agree/Disagree percentages, unique responses per prompt, and the percentage
   of complete prompt groups identical across every iteration.

## 10. Prepared seven-statement pilot

Prepared output directory:

`outputs/probvaa_authoritative_pilot_20261019_official_7statements/`

| Item | Value |
|---|---:|
| Underlying statements | First 7 groups in authoritative CSV order |
| Formulation rows | 28 |
| Personal templates | 6 |
| Iterations | 30 |
| Expected generations | 5,040 |
| Worker 0 tasks | 2,520 |
| Worker 1 tasks | 2,520 |
| Unique generation IDs | 5,040 |

- Manifest: `outputs/probvaa_authoritative_pilot_20261019_official_7statements/tasks.jsonl`
- Metadata: `outputs/probvaa_authoritative_pilot_20261019_official_7statements/metadata.json`
- Manifest SHA-256: `5e43b07b22b9b34b01a6fd95ff310745ead7c9273178724bfabb3ce545f2be33`
- Metadata SHA-256: `5e6d46a41365fc1a040cc596c3f23525bc01371342da56acce775b9c6addb658`

No pilot worker responses have been generated. The separately executed smoke
test generated one transient response and did not write to either worker file.

## 11. Pilot launch commands — executed after explicit approval

The commands below use verified physical GPU IDs 0 and 1. They were executed
only after explicit user approval and a repeated idle-GPU check.

Terminal/process 1:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_authoritative.py worker --manifest outputs/probvaa_authoritative_pilot_20261019_official_7statements/tasks.jsonl --model-key official --worker-id 0 --world-size 2 --output outputs/probvaa_authoritative_pilot_20261019_official_7statements/worker-0.jsonl
```

Terminal/process 2:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 envs/classifier/bin/python -u code/evaluation/run_probvaa_authoritative.py worker --manifest outputs/probvaa_authoritative_pilot_20261019_official_7statements/tasks.jsonl --model-key official --worker-id 1 --world-size 2 --output outputs/probvaa_authoritative_pilot_20261019_official_7statements/worker-1.jsonl
```

Only after both workers exit successfully:

```bash
envs/classifier/bin/python code/evaluation/run_probvaa_authoritative.py merge --manifest outputs/probvaa_authoritative_pilot_20261019_official_7statements/tasks.jsonl --worker-output outputs/probvaa_authoritative_pilot_20261019_official_7statements/worker-0.jsonl --worker-output outputs/probvaa_authoritative_pilot_20261019_official_7statements/worker-1.jsonl --output-dir outputs/probvaa_authoritative_pilot_20261019_official_7statements/merged
```

The merge output will contain `responses.jsonl`, an upstream-compatible
`responses.csv`, and `summary.json`. The summary reports Agree, Disagree, NA,
missing and failed counts; class percentages; NA breakdowns by template,
statement, and formulation; unique response counts for each prompt; and the
percentage of complete prompts with identical responses across all 30
iterations.

## 12. Preflight status and remaining uncertainties

| Check | Status |
|---|---|
| Authoritative commit and clean clone | PASS |
| Statement/template source data | PASS |
| Exact personal-template filter | PASS |
| Isolated `map_response` parity | PASS |
| CPU test suite | PASS |
| Official checkpoint files | PASS |
| Six base-plus-adapter artifact sets | PASS |
| Offline Python ML runtime | PASS (`envs/classifier`: Torch 2.11.0+cu130, Transformers 5.10.2, PEFT 0.19.1, Accelerate 1.13.0) |
| Static config/tokenizer/PEFT compatibility | PASS |
| GPU visibility and existing-process audit | PASS: nine RTX A6000 GPUs visible; GPUs 0/1 each 49,140 MiB total, 48,592 MiB free, 1 MiB used, 0% utilization; no compute processes listed before or after smoke test |
| `torch.cuda` access | PASS independently for physical GPUs 0 and 1; each process saw exactly one RTX A6000 |
| Per-GPU memory fit | PASS: smoke peak 14,664,257,536 bytes versus 50,952,536,064 bytes total |
| Exact physical GPU IDs | PASS: 0 and 1 |
| Runtime model-load/generation smoke test | PASS on GPU 0: load 8.98 s, generation 0.85 s, raw response `Support`, mapped value `1` (Agree) |
| Post-smoke GPU/process audit | PASS: GPUs 0/1 returned to 1 MiB used; no compute processes listed |
| Worker output/resume state | PASS: no worker output files exist; each worker has exactly 2,520 pending tasks |

The smoke test reproduced the effective configuration `do_sample=True`,
`temperature=1.0`, `top_p=None`, and `max_new_tokens=10`, with seed 42 reset
immediately before generation. Transformers warned that `temperature=1.0` and
`top_p=None` are semantically inactive/default generation flags; it did not
disable sampling. No parameter was changed.

Remaining methodological uncertainty: newline interpretation in the missing
upstream `format_prompt`. The prepared pilot retains literal backslash-`n`
sequences under the documented default assumption.

## 13. Recommendation

The approved 5,040-response pilot completed successfully: 5,040 unique task
IDs, zero duplicates, zero missing or failed generations, and complete raw and
summary outputs. See `PROBVAA_AUTHORITATIVE_PILOT_RESULTS.md` for the results
and methodological concerns. The 172,080-response full evaluation remains out
of scope and was not launched.
