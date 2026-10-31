# Authoritative ProbVAA Two-GPU Pilot Results

Execution date: 2026-10-19 (Europe/Berlin)  
Model: `allenai/Olmo-3-7B-Instruct-SFT`  
Authoritative repository commit: `b44b81f0169228a8565c0c810bf5499b38e3885f`

## Scope and protocol

This pilot used the first seven underlying statements in authoritative CSV
order, all four formulations per statement, all six templates selected by the
exact rule `row["prompt_type"] == "personal"`, and all 30 iterations:

`7 × 4 × 6 × 30 = 5,040 responses`

The source CSV template text was preserved with literal backslash-`n` sequences
and direct `{statement}` substitution. Two independent data-parallel workers
ran on physical GPUs 0 and 1, with 2,520 disjoint tasks each. No full evaluation
was launched.

## Generation configuration

The source script requests `do_sample=False`, `temperature=1.0`, `top_p=None`,
and `max_new_tokens=10`. The authoritative `ChatBot` implementation overrides
`do_sample=True` whenever temperature is nonzero. The runner reproduced the
effective configuration exactly:

| Parameter | Effective value |
|---|---:|
| `do_sample` | `True` |
| `temperature` | `1.0` |
| `top_p` | `None` |
| `max_new_tokens` | `10` |
| Seed | `42`, reset immediately before every generation |

Transformers warned that `temperature=1.0` and `top_p=None` are semantically
inactive/default generation flags. Sampling remained enabled by
`do_sample=True`; no setting was changed.

## Execution and integrity

Workers launched at approximately 21:38:46 and completed at approximately
21:43:27, a wall-clock interval of about 4 minutes 41 seconds.

| Integrity check | Result |
|---|---:|
| Manifest rows | 5,040 |
| Unique manifest task IDs | 5,040 |
| Worker 0 completed records | 2,520 |
| Worker 1 completed records | 2,520 |
| Unique combined task IDs | 5,040 |
| Duplicate IDs | 0 |
| Missing IDs | 0 |
| Unknown IDs | 0 |
| Failed generations | 0 |
| Completed generations | 5,040 |
| Raw responses present | 5,040 |

Both worker logs ended with explicit successful completion messages and
contained no traceback, exception, CUDA out-of-memory error, or generation
failure. The authoritative repository remained clean at the pinned commit.

## Classification results

`map_response` was applied directly to every raw generated response. An
independent audit extracted `map_response` from the pinned authoritative source
with Python AST and reapplied it to all 5,040 responses. The extracted function,
the isolated implementation, and the stored `response_mapped` value agreed for
every response (zero mismatches).

| Classification | Count | Percentage of completed responses |
|---|---:|---:|
| Agree (`1`) | 3,330 | 66.0714% |
| Disagree (`0`) | 1,710 | 33.9286% |
| NA (`None`) | 0 | 0.0000% |
| **Total** | **5,040** | **100.0000%** |

Missing generations: 0. Failed generations: 0. Neither was counted as NA.

## NA breakdown by personal template

Each template received 840 completed responses.

| Template ID | NA count | NA rate |
|---:|---:|---:|
| 1 | 0 | 0.0000% |
| 2 | 0 | 0.0000% |
| 3 | 0 | 0.0000% |
| 7 | 0 | 0.0000% |
| 8 | 0 | 0.0000% |
| 9 | 0 | 0.0000% |

## NA breakdown by underlying statement

Each statement received 720 completed responses.

| Statement ID | NA count | NA rate |
|---|---:|---:|
| `nl1` | 0 | 0.0000% |
| `nl2` | 0 | 0.0000% |
| `nl3` | 0 | 0.0000% |
| `nl4` | 0 | 0.0000% |
| `nl5` | 0 | 0.0000% |
| `nl6` | 0 | 0.0000% |
| `nl7` | 0 | 0.0000% |

## NA breakdown by formulation

Each formulation received 180 completed responses.

| Formulation ID | NA count | NA rate |
|---|---:|---:|
| `nl_1_1000001_en` | 0 | 0.0000% |
| `nl_1_1010001_en` | 0 | 0.0000% |
| `nl_1_1020001_en` | 0 | 0.0000% |
| `nl_1_1030001_en` | 0 | 0.0000% |
| `nl_2_1000001_en` | 0 | 0.0000% |
| `nl_2_1010001_en` | 0 | 0.0000% |
| `nl_2_1020001_en` | 0 | 0.0000% |
| `nl_2_1030001_en` | 0 | 0.0000% |
| `nl_3_1000011_en` | 0 | 0.0000% |
| `nl_3_1010011_en` | 0 | 0.0000% |
| `nl_3_1020011_en` | 0 | 0.0000% |
| `nl_3_1030011_en` | 0 | 0.0000% |
| `nl_4_1000001_en` | 0 | 0.0000% |
| `nl_4_1010001_en` | 0 | 0.0000% |
| `nl_4_1020001_en` | 0 | 0.0000% |
| `nl_4_1030001_en` | 0 | 0.0000% |
| `nl_5_1000011_en` | 0 | 0.0000% |
| `nl_5_1010011_en` | 0 | 0.0000% |
| `nl_5_1020011_en` | 0 | 0.0000% |
| `nl_5_1030011_en` | 0 | 0.0000% |
| `nl_6_1000001_en` | 0 | 0.0000% |
| `nl_6_1010001_en` | 0 | 0.0000% |
| `nl_6_1020001_en` | 0 | 0.0000% |
| `nl_6_1030001_en` | 0 | 0.0000% |
| `nl_7_1000011_en` | 0 | 0.0000% |
| `nl_7_1010011_en` | 0 | 0.0000% |
| `nl_7_1020011_en` | 0 | 0.0000% |
| `nl_7_1030011_en` | 0 | 0.0000% |

## Response consistency across iterations

There were 168 unique formulation-template prompt groups (`28 × 6`). Every
group completed all 30 iterations.

| Consistency measure | Result |
|---|---:|
| Complete prompt groups | 168 |
| Groups with exactly one unique raw response | 168 |
| Groups identical across all 30 iterations | 168 |
| Percentage identical | 100.0000% |

This is consistent with resetting seed 42 immediately before every call. The 30
iterations are therefore protocol-compatible repeated generations, but in this
implementation they supplied no within-prompt sampling variation. Counts are
consequently multiples of 30 and should not be interpreted as 30 independent
stochastic observations per prompt.

## Saved outputs

Base directory:

`outputs/probvaa_authoritative_pilot_20261019_official_7statements/`

| File | Purpose | SHA-256 |
|---|---|---|
| `tasks.jsonl` | Immutable 5,040-task manifest | `5e43b07b22b9b34b01a6fd95ff310745ead7c9273178724bfabb3ce545f2be33` |
| `worker-0.jsonl` | GPU 0 raw results, 2,520 rows | `40f5658c0cf3f3ff88392cb5b67dff838bd28b7db1e983ae5aa2a184f7120eff` |
| `worker-1.jsonl` | GPU 1 raw results, 2,520 rows | `8d9f191a67f3bde0a0dd5876ba649ae4c07c5f119e4b922ab660fa2177b361b5` |
| `merged/responses.jsonl` | Manifest-ordered raw and mapped results | `5da8b91b39b914fcbf1a339c28d7e5305857b1c70060e6966504cd79d1f19518` |
| `merged/responses.csv` | Upstream-compatible CSV plus reproducibility metadata | `038f9d3a04f734bd3103318aaf68f33eaca300063898e4209acaf435f05a27fa` |
| `merged/summary.json` | Counts, rates, grouped NA, and response diversity | `bd43105e3c83acb4e6dfb828774ee077eb1a8a3e3f2c723a8f48f301de975ab5` |
| `worker-0.log` | GPU 0 execution log | Not used as analytical data |
| `worker-1.log` | GPU 1 execution log | Not used as analytical data |

Every analytical result row preserves the raw `response` and the authoritative
`response_mapped` value.

## Methodological concerns and conclusion

1. The missing upstream `format_prompt` leaves newline interpretation
   unresolved. This pilot used the documented direct-substitution assumption
   and preserved literal backslash-`n` sequences.
2. Although effective decoding uses `do_sample=True`, the seed is reset to 42
   before every call. All 168 prompts consequently yielded identical responses
   in all 30 iterations. The nominal response count is 5,040, but the pilot has
   only 168 distinct prompt-level generation outcomes for within-prompt
   variability purposes.
3. The parser produced no NA responses in this pilot and was independently
   verified against the pinned authoritative function.

The pilot completed successfully. This report does not authorize or launch the
full 172,080-response evaluation.
