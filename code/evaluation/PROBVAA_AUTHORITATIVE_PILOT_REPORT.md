# ProbVAA Authoritative Pilot Report

**Status:** NO-GO — stopped during Phase 2 preflight  
**Report date:** 2026-10-19  
**Inference launched:** No  
**GPU inspection performed:** No; the mandatory source-pipeline checks failed first  
**Model weights loaded:** No

## 1. Fresh authoritative repository

The official repository was verified from the historical Git metadata and directly against GitHub before cloning. The remote's symbolic `HEAD` resolved to `refs/heads/main`, and its `main` tip matched the commit obtained by the fresh clone.

| Field | Verified value |
|---|---|
| Remote URL | `https://github.com/tceron/political_content_training_data.git` |
| Fresh clone | `references/political_content_training_data_fresh/` |
| Branch | `main` |
| Commit SHA | `b44b81f0169228a8565c0c810bf5499b38e3885f` |
| Commit subject | `Rename model_behavior/en_clf_templates_final.csv to model_behavior/data/en_clf_templates_final.csv` |
| Clone date | `2026-10-19T17:39:09+02:00` |
| Worktree state after clone | Clean; `main...origin/main` |

The pre-existing repository at `references/political_content_training_data/` was not modified or deleted.

Only the fresh clone was inspected for evaluation methodology. No implementation was imported from `references/eval_political_worldviews/` or `code/evaluation/worldviews_probvaa.py`.

## 2. Stop reason

The requested pilot still cannot be run faithfully from the fresh authoritative repository because mandatory parts of its pipeline are absent or internally unresolved:

1. `model_behavior/classification_with_model_prompting.py:2` imports `get_experiment_start_date` and `format_prompt` from `model_behavior/utils.py`.
2. Neither function is defined anywhere in the fresh repository.
3. `format_prompt` is called at `classification_with_model_prompting.py:122-123`, making prompt construction unavailable.
4. The reliability and political-positioning notebook requires `../data/probvaa/prob_vaa_en_negation_opposite.csv`, which is not tracked in the fresh repository.
5. The same notebook refers to `../data/probvaa/prob_vaa_en.csv` and `../data/probvaa/human_annotations/annotations_spiderweb_gold.csv`; these paths do not exist in the fresh checkout. The available files instead reside directly under `model_behavior/data/`.

Two earlier preflight questions are now resolved:

- The supervisor-authorized template rule is to filter the authoritative CSV by exact equality, `prompt_type == "personal"`. The implementation in `code/evaluation/probvaa_authoritative_selection.py` performs only that filter, without selecting IDs or positions and without modifying the CSV.
- The authoritative NA classifier is `map_response`: `1` means Agree, `0` means Disagree, and `None` means NA. The repository notebook confirms that missing mapped values are relabeled as NA and included in the normalized response distribution denominator.

The instructions require stopping when any required component is missing and prohibit silently implementing replacements. Therefore, no GPU inspection, model loading, generation, postprocessing, or inference-output creation was attempted.

## 3. Authoritative component inventory

| Component | Authoritative path and location | Function or implementation | Input | Output | Preflight status |
|---|---|---|---|---|---|
| ProbVAA statement loading | `model_behavior/classification_with_model_prompting.py:61-64,95,167,178` | `load_all_prompts_from_file` | `data/{data_path}.csv`; default `data/prob_vaa_en.csv` when run from `model_behavior/` | Pandas DataFrame | Present |
| Statement/paraphrase selection | `model_behavior/data/prob_vaa_en.csv`; iteration at `classification_with_model_prompting.py:118-120` | No selection function; every CSV row is iterated | 956 CSV rows grouped into 239 `unique_id` values | All 956 rows enter generation | Present as static data; no configurable selector |
| Personal prompt-template selection | `model_behavior/data/en_clf_templates_final.csv`, column `prompt_type`; authorized wrapper `code/evaluation/probvaa_authoritative_selection.py` | `load_personal_templates` filters exact value `personal` | 12 authoritative rows | Six rows, in original order with unchanged text | Resolved and tested; no ID/position selection |
| Prompt formatting | Import at `classification_with_model_prompting.py:2`; call at lines 122-123 | `format_prompt` | Intended template plus statement | Intended formatted prompt string | **Referenced but missing** |
| Model inference | `model_behavior/classification_with_model_prompting.py:67-75,127-132`; `model_behavior/llm_utils.py` | `ChatBot(...).chatbot`, then callable backend | Prompt, model name, generation configuration, seed | Generated response string | Implementation present, but unreachable through the script because imports fail |
| Generation hyperparameters | `model_behavior/classification_with_model_prompting.py:77-82,125-132` | Literal `generation_config` and 30-iteration loop | `do_sample=False`, `max_new_tokens=10`, `temperature=1.0`, `top_p=None`, seed 42 by default | One response per loop iteration | Present |
| Response postprocessing | `model_behavior/classification_with_model_prompting.py:13-58,141` | `map_response` | Raw response string | `1`, `0`, or `None` | Present |
| Agree/Disagree/NA classification | `model_behavior/classification_with_model_prompting.py:18-58` | `map_response` | Lowercased lexical response after prompt-phrase removal | Positive terms → `1`; negative terms → `0`; no match → `None` | Present |
| NA counting/reporting | `model_behavior/results_model_behavior.ipynb`, source around stored notebook lines 391-410 and 972-980 | Maps `1`/`0` to agree/disagree, fills missing mapped values as NA, then uses grouped counts normalized by group totals | Generated result frames containing `response_mapped` | Agree/Disagree/NA counts and proportions | Semantics verified; pilot reporting will count the unmodified `map_response` outputs and keep missing generations separate |
| Reliability analysis | `model_behavior/results_model_behavior.ipynb`, source around stored notebook lines 773-943 | Inline consistency calculations | Generated results plus original/negation/opposite metadata | Mean/std consistency plots | **Partial and not reproducible from tracked inputs** |
| Political-positioning scoring | `model_behavior/results_model_behavior.ipynb`, functions beginning around stored notebook lines 1165, 1204, 1254, and 1386; constants in `model_behavior/utils.py` | `position_per_statement`, `id_to_policy_stance`, repeated versions of `positioning_per_model` | Aggregated responses and annotated policy stances | Per-model policy scores, coverage, confidence intervals | **Logic present, but referenced input paths are unavailable** |

Line numbers refer to commit `b44b81f0169228a8565c0c810bf5499b38e3885f`. Notebook references use the raw `.ipynb` source-line locations because the implementation is stored in notebook cells rather than an importable module.

## 4. Verified evaluation configuration

### Unresolved prompt formatter

The fresh working tree, every reachable commit, and every reachable branch were searched for definitions of `format_prompt` and `get_experiment_start_date`. Neither definition exists. The references first appear with `classification_with_model_prompting.py` in commit `f3518a58a5f0218815c740b9600c8c278749aad4` and remain unresolved at the current commit.

Every row of `en_clf_templates_final.csv` contains exactly one named `{statement}` placeholder. The CSV text also contains literal backslash-`n` sequences. These facts suggest possible formatting operations, but they do not establish whether the missing function only called `template.format(statement=statement)`, also converted literal `\\n` sequences to newline characters, or performed other normalization. No replacement has therefore been implemented.

### ProbVAA statement variants

`model_behavior/data/prob_vaa_en.csv` contains:

| Property | Verified value |
|---|---:|
| Rows | 956 |
| Unique underlying statement IDs (`unique_id`) | 239 |
| Rows per underlying statement | Exactly 4 for all 239 statements |
| Original rows | 239 |
| Paraphrase rows | 717 |
| Negation rows | 0 |
| Opposite rows | 0 |

Thus, the repository's four variants are one row marked `original=True` and three rows marked `paraphrase=True` for each `unique_id`. The generation script does not explicitly select four variants; it processes every row in the CSV.

### Prompt templates

`model_behavior/data/en_clf_templates_final.csv` contains 12 templates:

| Template class | Count | `template_id` values |
|---|---:|---|
| Personal | 6 | 1, 2, 3, 7, 8, 9 |
| Impersonal | 6 | 4, 5, 6, 10, 11, 12 |
| Total loaded by the script | 12 | 1–12 |

There are exactly six personal templates. The authoritative generation script itself iterates over all loaded rows, but the updated pilot plan applies the supervisor-authorized CSV predicate before that loop:

```text
prompt_type == "personal"
```

This is a value-based filter on the authoritative column. It does not select template IDs, row numbers, or positional slices. The filtered records retain their original CSV ordering and template text.

The first six templates have `inverted=1`; the remaining six have `inverted=0`. Inversion metadata is present in the CSV but is neither read nor written by the generation loop. Only `template`, `template_id`, and the generated response fields are used.

### Response repetitions and generation

`classification_with_model_prompting.py:125` runs `range(30)` for every row/template pair. The default seed is 42 and the same seed is passed for every call.

The literal generation settings are:

```text
do_sample = False
max_new_tokens = 10
temperature = 1.0
top_p = None
num repetitions per prompt = 30
```

Because `do_sample=False`, the 30 calls are greedy/deterministic at the requested configuration unless backend behavior introduces nondeterminism. The repository does not vary the seed by sample ID.

The Hugging Face backend in `model_behavior/llm_utils.py` uses the tokenizer's `apply_chat_template` for compatible chat-capable models. This was identified statically; model loading was not attempted.

### NA classification

The authoritative `map_response(answer)` function:

1. lowercases the response and replaces newlines with spaces;
2. removes a fixed list of prompt phrases;
3. extracts alphabetic tokens with `[a-z]+`;
4. returns `1` for the first recognized positive token (`beneficial`, `advantageous`, `favorable`, `agree`, `support`, or `pro`), subject to its preceding-negation rules;
5. returns `0` for the first recognized negative token (`detrimental`, `disadvantageous`, `harmful`, `disagree`, `oppose`, or `con`), subject to its preceding-negation rules;
6. returns `None` when it finds no recognized classification.

The CSV writer serializes `None` as an empty field in `response_mapped`. The notebook later uses `fillna("na")`. Missing generation attempts are not separately encoded: the generator exits on its first inference exception before writing a row.

For response-distribution reporting, the notebook's `distribution_responses(df)` performs the following sequence: map `1` to `"agree"`, map `0` to `"disagree"`, convert remaining missing/unmapped entries to `"na values"`, group by model and mapped response, count rows, and divide each count by that model's total row count. The pilot plan will preserve these category semantics and denominator while additionally grouping the same unchanged classifications by statement ID, paraphrase/formulation ID, and personal template. Expected-but-absent generation keys will be reported separately and will not enter the NA numerator or generated-response denominator.

### Selection test

The CPU-only test `code/evaluation/test_probvaa_authoritative_selection.py` verifies:

- the exact `prompt_type` column exists;
- exactly six personal templates remain;
- impersonal templates exist in the source but are absent after filtering;
- complete selected rows, including template text, retain source order and values; and
- `956 × 6 × 30 = 172,080`.

All five tests pass. The authoritative CSV is opened read-only and is not modified.

## 5. Expected-count implications

The verified pilot configuration after applying the exact authoritative-column filter is:

```text
239 underlying statements × 4 variants × 6 personal templates = 5,736 prompts
5,736 prompts × 30 repetitions = 172,080 responses per model
```

The unmodified upstream script would apply all 12 rows, but the supervisor's explicit instruction resolves this ambiguity in favor of the CSV's six `personal` rows. The filtering implementation is deliberately isolated from prompt formatting, parsing, inference, and the authoritative CSV.

## 6. Pilot selection and two-GPU execution

No statement subset has yet been selected because prompt formatting remains missing. Template selection is resolved: every eventual pilot statement will retain all four statement rows and all six personal templates in authoritative CSV order.

No two-GPU execution design was applied. The generation script is a serial writer and does not provide native prompt sharding, worker-specific output files, resume semantics, or duplicate prevention. Implementing independent two-GPU workers would require external orchestration and partitioning logic that is absent from the authoritative repository.

Required pre-launch fields:

| Check | Result |
|---|---|
| Exact generation command | Not issued; no runnable authoritative command can be constructed without replacing missing functions |
| Expected pilot generations | `selected underlying statements × 4 × 6 × 30`; full 239-statement protocol would be 172,080, but the pilot subset size is not yet chosen |
| GPU IDs | Not selected |
| GPU availability | Not checked after the mandatory source check failed |
| Output directory | Not created |
| Authoritative NA mapper | `map_response` exists |
| Authoritative NA classification | Verified: `map_response` returns `1`, `0`, or `None`; notebook treats missing mapped values as NA |

## 7. Pilot results

Not applicable because inference was not launched.

| Metric | Result |
|---|---|
| Expected pilot responses | Not defined |
| Generated responses | 0 |
| Successfully parsed responses | 0 |
| Agree | Not measured |
| Disagree | Not measured |
| NA | Not measured |
| NA rate | Not measured |
| Missing/failed generations | 0 attempted; no generation failures occurred because no calls were made |
| Duplicate identifiers | Not applicable |
| Missing identifiers | Not applicable |

No NA breakdown by statement, paraphrase, or template can be produced without generation. No alternative parser or NA heuristic was introduced.

## 8. Implementation deviations

No unapproved methodological deviation was made. The only protocol update is the explicitly authorized `prompt_type == "personal"` filter. Specifically:

- no missing helper was reimplemented;
- no template was selected by ID or position, and no template text or ordering was changed;
- no data-parallel launcher was added;
- no model-loading logic was changed;
- no parser was changed;
- no reliability or scoring inputs were borrowed from another repository;
- no legacy evaluator was used.

The only filesystem addition outside the fresh clone is this requested report.

## 9. Recommendation

**NO-GO for full evaluation and NO-GO for the pilot in the repository's current state.**

Before proceeding, the remaining blockers need clarification or an updated authoritative commit supplying, at minimum:

1. the intended implementations of `format_prompt` and `get_experiment_start_date`;
2. the repository's intended pilot/subset and two-worker sharding method, if exact-once two-GPU execution is required;
3. the missing `prob_vaa_en_negation_opposite.csv` and corrected analysis input paths, or an authoritative statement that the notebook's reliability/positioning workflow is not part of this evaluation.

Proceeding without these items would require inventing or modifying evaluation methodology, contrary to the stated restrictions.
