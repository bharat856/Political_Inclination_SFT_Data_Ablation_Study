# ProbVAA Seed-Handling Pilot Comparison

Execution date: 2026-10-19 (Europe/Berlin)  
Model: `allenai/Olmo-3-7B-Instruct-SFT`  
Authoritative reference commit: `b44b81f0169228a8565c0c810bf5499b38e3885f`

## Diagnostic question

The authoritative behavior resets seed 42 immediately before every
generation. The first pilot therefore produced one raw response per prompt,
repeated 30 times. The second pilot changed only seed handling: each GPU worker
was seeded once with 42 after model loading, after which its RNG state advanced
without per-generation reseeding.

`reset_each_generation` remains the default evaluation mode. The diagnostic
did not modify the authoritative repository, prompt formatting, chat template,
model, generation parameters, classifier, statement/formulation selection,
template selection, task IDs, ordering, or shards.

## Identical evaluation inputs

Both pilots used the exact same original manifest:

`outputs/probvaa_authoritative_pilot_20261019_official_7statements/tasks.jsonl`

Manifest SHA-256:
`5e43b07b22b9b34b01a6fd95ff310745ead7c9273178724bfabb3ce545f2be33`

| Quantity | Both pilots |
|---|---:|
| Underlying statements | 7 |
| Formulations per statement | 4 |
| Personal templates | 6 |
| Iterations per prompt | 30 |
| Prompt groups | 168 |
| Generation tasks | 5,040 |
| Worker 0 tasks | 2,520 |
| Worker 1 tasks | 2,520 |

Effective generation configuration in both pilots was `do_sample=True`,
`temperature=1.0`, `top_p=None`, and `max_new_tokens=10`. Literal prompt
substitution and the authoritative `map_response` heuristic were unchanged.

## Main results

| Measure | Reset each generation | Seed once | Difference |
|---|---:|---:|---:|
| Completed generations | 5,040 | 5,040 | 0 |
| Agree | 3,330 (66.0714%) | 3,172 (62.9365%) | −158 (−3.1349 pp) |
| Disagree | 1,710 (33.9286%) | 1,840 (36.5079%) | +130 (+2.5794 pp) |
| NA | 0 (0.0000%) | 28 (0.5556%) | +28 (+0.5556 pp) |
| Missing generations | 0 | 0 | 0 |
| Failed generations | 0 | 0 | 0 |
| Duplicate task IDs | 0 | 0 | 0 |
| Prompts identical across all 30 iterations | 168/168 (100.0000%) | 7/168 (4.1667%) | −161 prompts |
| Average unique raw responses per prompt | 1.0000 | 3.2202 | +2.2202 |

Seed-once unique-response counts ranged from 1 to 8 per prompt:

| Unique raw responses | Number of prompts |
|---:|---:|
| 1 | 7 |
| 2 | 42 |
| 3 | 57 |
| 4 | 40 |
| 5 | 16 |
| 6 | 4 |
| 7 | 1 |
| 8 | 1 |

## Examples of prompts with varied responses

These are exact raw-response variants; capitalization, punctuation, and
continuations are therefore counted separately.

| Formulation | Template | Statement | Unique responses | Selected observed variants |
|---|---:|---|---:|---|
| `nl_5_1010011_en` | 1 | Motorists should be taxed based on kilometers driven, not car ownership. | 8 | `support` (12), `oppose` (6), `Oppose.` (2), `Support.` (2), `Support` (2), plus three longer/case variants |
| `nl_5_1000011_en` | 8 | Replace car-ownership tax with tax per kilometer driven. | 7 | `beneficial` (10), `harmful` (10), `Beneficial` (2), plus four longer variants |
| `nl_5_1000011_en` | 2 | Replace car-ownership tax with tax per kilometer driven. | 6 | `beneficial` (20), `harmful` (2), `Harmful` (2), plus three longer variants |
| `nl_6_1010001_en` | 2 | Permit celebratory fireworks again around New Year's Eve. | 6 | `harmful` (12), `beneficial` (10), `Harmful` (2), plus three longer variants |
| `nl_5_1030011_en` | 9 | Replace car-ownership tax with a distance-based driving tax. | 6 | `favorable` (18), `Favorable` (4), `Favorable.` (2), `favorable.` (2), `detrimental` (2), and one longer detrimental response |

## NA examples

NA means exactly that the unchanged authoritative `map_response` function
returned `None`. Missing and failed generations are not NA. Examples include:

| Formulation | Template | Sample | Exact raw response |
|---|---:|---:|---|
| `nl_7_1020011_en` | 1 | 0 | `Depends. There are several factors: (What` |
| `nl_7_1030011_en` | 1 | 6 | ` supportive` |
| `nl_3_1000011_en` | 2 | 4 | `Benefits: Childcare for all parents would lead to` |
| `nl_4_1000001_en` | 2 | 2 | `benign` |
| `nl_1_1020001_en` | 2 | 8 | `Let me think through this step by step. The` |
| `nl_1_1010001_en` | 9 | 26 | `detimental` |

All 28 NA responses were preserved verbatim. The authoritative mapper extracted
directly from the pinned source reproduced every stored mapping with zero
mismatches.

## Seed-once NA distribution by template

Each template has 840 completed responses.

| Template ID | NA | NA rate |
|---:|---:|---:|
| 1 | 4 | 0.4762% |
| 2 | 8 | 0.9524% |
| 3 | 8 | 0.9524% |
| 7 | 0 | 0.0000% |
| 8 | 0 | 0.0000% |
| 9 | 8 | 0.9524% |

## Seed-once NA distribution by statement

Each statement has 720 completed responses.

| Statement | NA | NA rate |
|---|---:|---:|
| `nl1` | 6 | 0.8333% |
| `nl2` | 2 | 0.2778% |
| `nl3` | 6 | 0.8333% |
| `nl4` | 6 | 0.8333% |
| `nl5` | 2 | 0.2778% |
| `nl6` | 2 | 0.2778% |
| `nl7` | 4 | 0.5556% |

## Seed-once NA distribution by formulation

Each formulation has 180 completed responses.

| Formulation | NA | NA rate |
|---|---:|---:|
| `nl_1_1000001_en` | 0 | 0.0000% |
| `nl_1_1010001_en` | 4 | 2.2222% |
| `nl_1_1020001_en` | 2 | 1.1111% |
| `nl_1_1030001_en` | 0 | 0.0000% |
| `nl_2_1000001_en` | 2 | 1.1111% |
| `nl_2_1010001_en` | 0 | 0.0000% |
| `nl_2_1020001_en` | 0 | 0.0000% |
| `nl_2_1030001_en` | 0 | 0.0000% |
| `nl_3_1000011_en` | 4 | 2.2222% |
| `nl_3_1010011_en` | 2 | 1.1111% |
| `nl_3_1020011_en` | 0 | 0.0000% |
| `nl_3_1030011_en` | 0 | 0.0000% |
| `nl_4_1000001_en` | 4 | 2.2222% |
| `nl_4_1010001_en` | 2 | 1.1111% |
| `nl_4_1020001_en` | 0 | 0.0000% |
| `nl_4_1030001_en` | 0 | 0.0000% |
| `nl_5_1000011_en` | 2 | 1.1111% |
| `nl_5_1010011_en` | 0 | 0.0000% |
| `nl_5_1020011_en` | 0 | 0.0000% |
| `nl_5_1030011_en` | 0 | 0.0000% |
| `nl_6_1000001_en` | 0 | 0.0000% |
| `nl_6_1010001_en` | 0 | 0.0000% |
| `nl_6_1020001_en` | 2 | 1.1111% |
| `nl_6_1030001_en` | 0 | 0.0000% |
| `nl_7_1000011_en` | 0 | 0.0000% |
| `nl_7_1010011_en` | 0 | 0.0000% |
| `nl_7_1020011_en` | 2 | 1.1111% |
| `nl_7_1030011_en` | 2 | 1.1111% |

The reset-each-generation pilot had zero NA responses in every template,
statement, and formulation.

## Execution integrity and runtime

| Check | Seed-once result |
|---|---:|
| Worker 0 records | 2,520 |
| Worker 1 records | 2,520 |
| Combined unique task IDs | 5,040 |
| Duplicate IDs | 0 |
| Missing IDs | 0 |
| Unknown IDs | 0 |
| Completed statuses | 5,040 |
| Failed statuses | 0 |
| Raw responses present | 5,040 |
| Source-extracted mapper mismatches | 0 |

The workers started at approximately 22:14:44 and finished at approximately
22:19:53, a wall-clock runtime of about 5 minutes 8 seconds. Both logs ended
with explicit completion messages and contained no traceback, exception, CUDA
out-of-memory error, or worker failure.

## Two-worker RNG-stream limitation

Both workers were initialized with seed 42, as documented in diagnostic
metadata. Because modulo-two sharding assigns adjacent sample IDs for the same
prompt to different workers, the two workers traversed the same prompt sequence
with identical RNG streams:

- 2,520/2,520 worker-position pairs had the same prompt.
- 2,520/2,520 pairs represented adjacent sample IDs.
- 2,520/2,520 pairs produced identical raw responses.

The diagnostic therefore provided at most 15 distinct stochastic draws per
prompt, duplicated across the two GPUs, rather than 30 independent draws. This
does not invalidate the finding that per-generation reseeding caused the first
pilot's complete repetition, but it means the observed seed-once diversity is a
conservative estimate of what distinct per-worker streams might yield. No
additional evaluation was launched to address this limitation.

## Reproducibility and resume behavior

`reset_each_generation` remains the default and retains the original behavior.
`seed_once` initializes after model loading and performs no seed call inside
`generate()`. Both diagnostic workers used seed 42. Reproduction therefore
requires the same two-worker shard mapping, task order, library stack, and
hardware behavior.

Worker output remains append-only and duplicate-safe. The implementation now
requires existing records to be the exact ordered prefix of the assigned shard.
For future interrupted seed-once runs it atomically checkpoints and restores
Python, NumPy, CPU Torch, and CUDA RNG state after each completed generation,
refusing a mismatched output/checkpoint instead of silently restarting the RNG
stream. This diagnostic completed without interruption.

## Output files

New diagnostic directory:

`outputs/probvaa_authoritative_pilot_seed_once_20261019/`

| File | SHA-256 |
|---|---|
| `metadata.json` | `afb6d9a34dee5c3a1f72836e919ddaeec2edf200c9b3f67aea68361dc6ddc6de` |
| `worker-0.jsonl` | `ef1c31b9f268dcdb11139433f8d089e39ade30885f08d646503c00c7fb682e27` |
| `worker-1.jsonl` | `eef424f42ff50da0de67e4cd7e33c287a242bf221bc887853b7b97c5da08f3d1` |
| `merged/responses.jsonl` | `3328927b756e30513bafbf85ebffcfd11f49a1e385b2b88ba129254a5e17b997` |
| `merged/responses.csv` | `16d9cdc5cb6d92e88176d115b906b96d8df4a9d9f01e4e5066e2b57ec3f1b5b0` |
| `merged/summary.json` | `3e83811d65e47f041a5b107898b6ab7c8146b06706e6d6d8291bddd500eda116` |

Raw responses and mapped labels are retained in both worker JSONL files and the
merged JSONL/CSV. Seed mode, worker seed, and worker ID are stored on every new
result row.

All nine original pilot file hashes were unchanged after the diagnostic. The
authoritative reference repository remained clean at the pinned commit.

## Conclusion

Response diversity clearly increased after removing the per-generation seed
reset: identical prompts fell from 100.0000% to 4.1667%, while average unique
raw responses rose from 1.0000 to 3.2202. NA frequency increased from 0% to
0.5556% because the advancing RNG exposed low-probability responses outside
the authoritative keyword set.

Because seed handling was the only intended methodological change and the same
manifest and inference settings were used, the results strongly support the
conclusion that resetting seed 42 before every generation was responsible for
the first pilot's identical repetitions. The identical two-worker RNG streams
mean this diagnostic still understates potential response diversity and should
be described as a controlled seed-handling diagnostic, not as a replacement
for the authoritative evaluation methodology.

No full evaluation and no additional diagnostic evaluation were launched.
