# Entire Project Context: Causal Political Content

Prepared: 30 October 2026. This is a standalone handoff for a collaborator or an AI assistant. It consolidates local project documentation and retained reports; it does not include raw datasets, model weights, environments, or every source file. No training, inference, downloads, or external verification were performed to produce this handoff. Paths are relative to the project root unless stated otherwise.

## Read this first: current state and precedence

The project investigates whether removing or balancing political and ideological content in supervised fine-tuning (SFT) data changes an LLM's political responses while preserving instruction-following ability. The work progressed from political classification benchmarks to target-domain detection, corpus interventions, six completed OLMo 3 QLoRA fine-tunes, instruction-following evaluation, political-evaluation diagnostics, and a corrected six-model political response collection.

**Latest verified milestone in retained reports:** the corrected ProbVAA evaluation completed for all six intervention models on 21 October 2026, producing 1,032,480 records. The report explicitly says no political-positioning scores were calculated in that execution. Agreement rates are not ideology scores or evidence of causal significance.

Documentation contains historical states. Resolve them in this order:

1. Corrected full-run final execution report: completed, verified six-model response collection.
2. Controlled Full diagnostic and remaining-model diagnostic: evidence for training-aligned input serialization.
3. End-to-end thesis audit: independently checked training/data defects, original-evaluation integrity, and methodological limitations. Its recommendations to hold diagnostics/evaluations describe the state BEFORE the subsequent diagnostic and corrected run.
4. October 19 dataset and fine-tuning audits: authoritative final six-condition counts and training settings.
5. October 11–12 evaluation progress: authoritative completed IFEval results, but its political-evaluation NO-GO and unlaunched status are historical and superseded by later reports.
6. Benchmark summary, chronological README, presentation report, and old project_context_prompt.txt: historical development, not final experiment status.

These are artifact-backed historical statuses, not a live process/GPU inspection. Do not infer current resource availability from saved PID or GPU tables.

## Research design and essential terminology

The final training source is ONLY `allenai/Dolci-Instruct-SFT-7B`, containing 2,152,112 original conversations/examples. Earlier corpus construction also handled `Dolci-Think-SFT-7B` (2,268,178 examples), yielding a combined 4,420,290 rows. Those combined counts MUST NOT be reported as the final six models' training dataset sizes. Earlier Tulu/OLMo-2 experiments are developmental predecessors.

Political detection is an OR union of a target-domain ModernBERT political gate and an MPNet embedding/logistic-regression gate at threshold 0.4. Instruct has 408,582 political candidates and 1,743,530 complement rows. Both gates select 78,539; MPNet alone selects 276,836; ModernBERT alone selects 53,207. These operational labels are not human ground truth.

Candidate slant labels combine Qwen/Gemma through the TANISE rule: keep agreement; slant beats neutral; Gemma breaks direct left/right disagreement; valid available predictions replace missing predictions. Instruct has 14,859 left, 2,940 right, and 390,783 neutral political rows, with no unresolved final labels. Neutral political is distinct from non-political. The slant prompt uses US-centered political definitions, which limits interpretation on multinational political questions.

| Condition | Examples | Definition | Optimizer steps |
|---|---:|---|---:|
| Full | 2,152,112 | All Instruct examples | 33,624 |
| NoPol | 1,743,530 | Remove all 408,582 political candidates | 27,240 |
| NoIdeology | 2,134,313 | Remove left and right; retain neutral political | 33,346 |
| NoLeft | 2,137,253 | Remove 14,859 left examples | 33,392 |
| NoRight | 2,149,172 | Remove 2,940 right examples | 33,578 |
| LRBalanced | 2,140,193 | Remove 11,919 left examples, retaining 2,940 per side | 33,438 |

Every condition streams the same flattened base file and excludes designated row IDs. Final interventions add no duplication or upsampling. Upstream semantic duplicates were not ruled out. The Instruct-specific LRBalanced sampling seed/procedure is not retained. Label IDs vary across tasks; consult each script rather than treating binary left/right IDs as political/non-political IDs.

## Final training configuration and limitations

Base: local `hf_cache/allenai_Olmo-3-1025-7B`, corresponding to `allenai/Olmo-3-1025-7B`. All six runs completed; terminal states, adapter artifacts, finite values, and local checksum manifests were audited. Base remote revision is not pinned. These are custom-format full-sequence QLoRA adaptations, not exact reproductions of official native-chat SFT.

Common settings: NF4 4-bit quantization with double quantization and BF16 compute; LoRA rank 16, alpha 32, dropout 0.05; attention q/k/v/o and MLP gate/up/down projections; 39,976,960 trainable LoRA parameters; frozen 7,298,011,136-parameter base. Sequence length 2,048; no packing; right dynamic padding; keep-start truncation. Full non-padding sequence loss includes user/system/assistant/role text. AdamW 8-bit, learning rate 2e-5, linear decay, 3% warmup, weight decay 0, clipping 1.0, seeds 42. Four ranks, per-device batch 1, accumulation 16, global effective batch 64. Approximately one pass per condition; absolute steps/token exposure differ, especially NoPol. This is a causal interpretation limitation.

Conversation serialization is plain text with actual newlines:

```text
USER:
{instruction}

ASSISTANT:
{response}
```

No native chat template was applied. Final trainer scripts are v4 for Full/NoLeft and v5 for the other four conditions. The final loader is `code/dolci_intervention_loader_instruct_v1.py`.

Confirmed serialization defects: 227,579 rows contain structured tool/function fields omitted from training text; 207,771 rows contain null assistant contents, producing 626,290 artificial `None` assistant messages; 11,723 rows have no real nonempty assistant content before truncation. Instruct source metadata was also lost because flattening looked for `dataset_source` rather than `source_dataset`/`domain`. Ordinary role boundaries were preserved. These limitations remain after fixing inference formatting. Finite updated adapters establish successful optimization, not equivalence to official SFT or validated political reliability. A systematic late loss increase is documented; it does not by itself establish divergence or overfitting.

## Evaluation: keep the protocols separate

### IFEval

Completed official scoring on 541 prompts / 834 constraints per model for four models, using greedy decoding, 1,280-token cap, training-aligned USER/ASSISTANT format, and seed 42.

| Model | Strict prompt | Loose prompt | Strict instruction | Loose instruction |
|---|---:|---:|---:|---:|
| Full | 56.56% | 60.63% | 66.79% | 70.02% |
| NoLeft | 57.49% | 62.11% | 66.67% | 70.50% |
| NoRight | 58.04% | 62.85% | 67.15% | 71.58% |
| LRBalanced | 54.71% | 58.41% | 64.39% | 67.99% |

Primary artifacts: `outputs/evaluation_recovery_20261011/final_comparison/`. Preserve the official scorer's documented punctuation-checker anomalies. Scores alone establish neither significance nor broad capability. No six-model IFEval completion is established by the inspected reports.

Independent official-checkpoint IFEval comparison lives in `validation/olmo3_sft_reproduction/`. Its retained comparison report still lists the official matched-local result as pending. Published official scores use different generation settings and cannot be treated as matched empirical comparisons. Some setup README statements about absent downloads may be stale; inspect files before assuming availability.

### Original ProbVAA and audit

The authoritative later reference snapshot is `references/political_content_training_data_fresh/`, recorded commit `b44b81f0169228a8565c0c810bf5499b38e3885f`. Do not silently substitute the older worldview repository or a different response mapper. The selected design uses 239 statements, four formulations and six personal templates: 5,736 prompt groups/model, repeated 30 times = 172,080 response records/model. This is a selected subset, not the entire original paper design.

The original audited collection contains 860,400 records across Official and four local variants. Bare prompts lack the custom adapters' learned assistant boundary. The audit found instruction continuations, high NA, and misleading lexical mapping hits. Those outputs remain useful evidence of protocol behavior but cannot be mixed with corrected outputs for substantive positioning claims.

Original mapping is lexical Agree/Disagree/NA. Notebook aggregation retains NA in the denominator; an all-NA item becomes zero, which is not evidence of genuine neutrality. Literal backslash-n sequences inside template bodies are preserved locally; the author's original formatter is unavailable, so equivalence is unresolved. Seed 42 resets before each generation: thirty nominal sample slots are not thirty independent random draws. The original audit found identical repetitions per group. Corrected runs preserve this seed policy.

### Controlled diagnostics and corrected full run

Full diagnostic: same 100 selected prompt groups, reused bare outputs versus 200 new aligned outputs at caps 10/32. NA fell from 65% to zero; all aligned outputs were complete stance words and terminated naturally within five tokens including EOS. All 100 aligned 10-token outputs and token sequences matched their 32-token counterparts. This supports the format correction without evidence for a larger cap on that sample.

Five remaining adapters were tested on the same 100 groups (500 new responses): all gave valid direct answers except one NoLeft `D` + EOS, preserved as NA. This was a selected diagnostic sample, not a population estimate.

The corrected full evaluation then completed for all six local models. Only the training-aligned outer wrapper and additive live logging changed. Inner literal prompt bodies, statement/formulation/template selection, original mapper, temperature 1, effective sampling, ten-token cap, and reset42 policy were preserved. Official was not rerun.

| Model | Agree | Disagree | NA | Total |
|---|---:|---:|---:|---:|
| Full | 76,560 | 95,520 | 0 | 172,080 |
| NoPol | 106,410 | 65,280 | 390 | 172,080 |
| NoIdeology | 94,890 | 77,190 | 0 | 172,080 |
| NoLeft | 65,100 | 105,960 | 1,020 | 172,080 |
| NoRight | 78,090 | 93,990 | 0 | 172,080 |
| LRBalanced | 78,390 | 93,690 | 0 | 172,080 |

Authoritative run directory: `outputs/probvaa_corrected_full_20261021/20261021T080830Z/`. Final report verifies exactly 1,032,480 records, no duplicate/missing keys, original mappings/wrappers, and unchanged checked reference/training/adapter/Official assets. Reboot recovery occurred; final verification establishes completion. This run calculated no political-positioning scores. Agree/Disagree describe the presented statement, not left/right ideology. Response coverage improvement does not alone establish political coherence, formulation/order robustness, or causal effects.

## Repository map and practical continuation

| Location | Role |
|---|---|
| `code/` | Dataset builders, gates, slant classification, interventions, training |
| `code/evaluation/` | IFEval, ProbVAA runners, parsing, scoring, protocol audits and tests |
| `data/raw/`, `data/processed/` | Original corpora, flattened conversations, benchmark/manual labels |
| `data/interventions/dolci_olmo3_sft_union_mpnet_t04/` | Final ensemble labels and removal row lists |
| `outputs/sft_runs/` | Final adapters/checkpoints and historical training experiments |
| `outputs/thesis_end_to_end_audit_20261021/` | Independent source/data/training/original-evaluation audit |
| `outputs/probvaa_*` | Distinct pilots, audits, diagnostics, original and corrected protocols |
| `references/` | Reference repositories; fresh snapshot is authoritative for later ProbVAA |
| `validation/olmo3_sft_reproduction/` | Isolated official-model instruction-following comparison |
| `apps/` | Manual SFT labeler and ModernBERT OOD error inspector |
| `scripts/`, `presentations/` | Setup/download helpers and presentation generation/artifacts |
| `logs/` | Historical run logs |
| `envs/`, `cache/`, `hf_cache/`, `models/` | Local environments, dependencies, datasets and model caches |

Main historical Python environment: `envs/classifier/bin/python`; run scripts from the project root. Dependencies differ across training, vLLM labeling, evaluation and R corpus processing. Check retained run metadata before reproducing. Many paths are machine-specific. No single universal dependency lock or full fresh-machine reproduction was established here.

Final run directories under `outputs/sft_runs/`: `olmo3_instruct_full_4gpu_v4`, `olmo3_instruct_noleft_4gpu_v4`, `olmo3_instruct_noright_4gpu_v5`, `olmo3_instruct_nopol_4gpu_v5`, `olmo3_instruct_noideology_4gpu_v5`, `olmo3_instruct_lrbalanced_4gpu_v5`.

Historical classifier findings: Manifesto chunk labels inherit document RILE scores and are noisy; performance drops on the larger benchmark. Cleaner external left/right benchmarks do not establish target-domain SFT accuracy. ModernBERT misses some neutral political OOD examples, motivating the embedding union. Full historical metrics and script workflow are included below.

Next scientific work should start from corrected full-run artifacts and an explicitly chosen scoring/reliability target. Preserve raw versus corrected protocol separation, account for repeated seeds, check formulation/template consistency, and disclose serialization and unequal-exposure limitations. Do not infer training-data causal effects directly from aggregate agreement. No further training, model downloads, GPU jobs, or scoring changes are authorized merely by this context file; historical launch commands are provenance, not instructions to execute.

## Embedded source documents

The following appendices preserve detailed methodology, historical classifier results, paths, evidence and caveats in this single file. They are source snapshots, not a unified chronology. Relative links inside them refer to their original source directory and may not resolve in this standalone copy. Use the stated source path to locate an artifact in the repository. The precedence and current-state summary above govern apparent contradictions.


---

# Appendix 1: Latest corrected execution

Source: `outputs/probvaa_corrected_full_20261021/20261021T080830Z/FINAL_EXECUTION_REPORT.md`.

# Corrected six-GPU ProbVAA execution report

Overall status: **COMPLETED AND VERIFIED**.

Start: 2026-10-21T08:10:53.271071+00:00
End: 2026-10-21T19:43:48.897061+00:00
Tmux session: `probvaa_corrected_6gpu` (left intact).
Output directory: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z`

Only the training-aligned input wrapper and additive live logging changed. Original instructions, literal newline behavior, statements, formulations, templates, ten-token sampled generation, seed 42 reset before every call, and original classifier were preserved. Official was not rerun. No political-positioning scores were calculated.

| Model | Agree | Disagree | NA | Total | GPU | PID | Status / verification |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 76,560 (44.4909%) | 95,520 (55.5091%) | 0 (0.0000%) | 172,080 | 0 | 3901588 | completed / True |
| NoPol | 106,410 (61.8375%) | 65,280 (37.9358%) | 390 (0.2266%) | 172,080 | 1 | 3901589 | completed / True |
| NoIdeology | 94,890 (55.1430%) | 77,190 (44.8570%) | 0 (0.0000%) | 172,080 | 2 | 3901590 | completed / True |
| NoLeft | 65,100 (37.8312%) | 105,960 (61.5760%) | 1,020 (0.5927%) | 172,080 | 3 | 3901591 | completed / True |
| NoRight | 78,090 (45.3801%) | 93,990 (54.6199%) | 0 (0.0000%) | 172,080 | 4 | 3901592 | completed / True |
| LRBalanced | 78,390 (45.5544%) | 93,690 (54.4456%) | 0 (0.0000%) | 172,080 | 5 | 3901593 | completed / True |

Total completed: 1,032,480 / 1,032,480.
Checked original reference/training/adapter/Official asset hashes unchanged: True.

## Model files and errors

- **Full**: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z/Full`; start `2026-10-21T08:10:54.337247+00:00`, end `2026-10-21T19:34:00.471806+00:00`; exit `0`; failure/interruption events `1`; error `None`.
  Files: [responses.jsonl](Full/responses.jsonl), [progress.log](Full/progress.log), [summary.json](Full/summary.json), [metadata.json](Full/metadata.json), [worker.log](Full/worker.log), [verification.json](Full/verification.json).
- **NoPol**: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z/NoPol`; start `2026-10-21T08:10:54.274028+00:00`, end `2026-10-21T18:47:06.268008+00:00`; exit `0`; failure/interruption events `1`; error `None`.
  Files: [responses.jsonl](NoPol/responses.jsonl), [progress.log](NoPol/progress.log), [summary.json](NoPol/summary.json), [metadata.json](NoPol/metadata.json), [worker.log](NoPol/worker.log), [verification.json](NoPol/verification.json).
- **NoIdeology**: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z/NoIdeology`; start `2026-10-21T08:10:54.355132+00:00`, end `2026-10-21T18:49:45.010131+00:00`; exit `0`; failure/interruption events `1`; error `None`.
  Files: [responses.jsonl](NoIdeology/responses.jsonl), [progress.log](NoIdeology/progress.log), [summary.json](NoIdeology/summary.json), [metadata.json](NoIdeology/metadata.json), [worker.log](NoIdeology/worker.log), [verification.json](NoIdeology/verification.json).
- **NoLeft**: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z/NoLeft`; start `2026-10-21T08:10:54.352618+00:00`, end `2026-10-21T19:43:45.078719+00:00`; exit `0`; failure/interruption events `1`; error `None`.
  Files: [responses.jsonl](NoLeft/responses.jsonl), [progress.log](NoLeft/progress.log), [summary.json](NoLeft/summary.json), [metadata.json](NoLeft/metadata.json), [worker.log](NoLeft/worker.log), [verification.json](NoLeft/verification.json).
- **NoRight**: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z/NoRight`; start `2026-10-21T08:10:54.374795+00:00`, end `2026-10-21T19:14:39.175363+00:00`; exit `0`; failure/interruption events `1`; error `None`.
  Files: [responses.jsonl](NoRight/responses.jsonl), [progress.log](NoRight/progress.log), [summary.json](NoRight/summary.json), [metadata.json](NoRight/metadata.json), [worker.log](NoRight/worker.log), [verification.json](NoRight/verification.json).
- **LRBalanced**: `/fs/scratch/users/bharatpi/causal_political_content/outputs/probvaa_corrected_full_20261021/20261021T080830Z/LRBalanced`; start `2026-10-21T08:10:54.321285+00:00`, end `2026-10-21T19:24:32.047083+00:00`; exit `0`; failure/interruption events `1`; error `None`.
  Files: [responses.jsonl](LRBalanced/responses.jsonl), [progress.log](LRBalanced/progress.log), [summary.json](LRBalanced/summary.json), [metadata.json](LRBalanced/metadata.json), [worker.log](LRBalanced/worker.log), [verification.json](LRBalanced/verification.json).

Per-model final verification streams every response against the authoritative task manifest, verifies original mappings and wrappers, exact 172,080 records, 5,736 groups, 30 distinct samples per group, no duplicates/missing keys, zero failed result records, and exhaustive Agree/Disagree/NA counts. Failure/interruption events are reported separately rather than recoded as NA.

Smoke outputs were retained as the first three records of each full evaluation; no extra smoke generations inflate the evaluation total. Both response and progress files are flushed during execution. Summary replacements are atomic. Single-writer locks and validated prefix resume prevent duplicate committed responses.

Reconnect: `tmux attach -t probvaa_corrected_6gpu`. Monitor per-model `progress.log` or `summary.json`. The tmux session is not terminated by the supervisor.


---

# Appendix 2: Full format diagnostic

Source: `outputs/probvaa_full_controlled_diagnostic_20261021/REPORT.md`.

# Full-only controlled ProbVAA diagnostic

**Training-aligned serialization removed NA on this 100-group sample. Increasing the token budget from 10 to 32 did not change any response.** These findings justify a corrected full evaluation using the adapters' actual training serialization; they provide no evidence that 32 tokens are needed for these prompts. No corrected full evaluation or other model run was launched.

## Original mapping results

All conditions use the same 100 prompt groups. Proportions include every selected group; no responses are filtered by the separate validity review.

| Condition | Input format | Maximum new tokens | Agree | Disagree | NA |
|---|---|---:|---:|---:|---:|
| A | Saved original bare prompt | 10 | 25/100 (25%) | 10/100 (10%) | 65/100 (65%) |
| B | Training-aligned role serialization | 10 | 46/100 (46%) | 54/100 (54%) | 0/100 (0%) |
| C | Same training-aligned role serialization | 32 | 46/100 (46%) | 54/100 (54%) | 0/100 (0%) |

These are the **unchanged original mapper's** Agree/Disagree/NA categories. The mapper's NA result is still `None`. Qualitative annotations below do not replace or alter those mappings.

## 1. Verified training serialization

`code/flatten_olmo3_dolci_sft.py::messages_to_text` uppercases each role, appends a colon and actual newline, appends stripped content, and joins messages with two actual newlines. `code/dolci_intervention_loader_instruct_v1.py::iter_split` passes `row_id`, `text`, and `split`, without structured messages. Full's `code/train_olmo3_lora_dolci_instruct_v4.py` consumes this `text` column through `SFTTrainer`, using the base tokenizer. It does not apply the official chat template.

The original formatter was loaded as an unmodified AST definition, without running the flattening script's data-loading top-level code. Its serialization was checked against saved training row 0: the full saved `text` exactly equals the formatter applied to that row's `messages`.

B and C therefore use:

```text
USER:
{unchanged original ProbVAA prompt}

ASSISTANT:
```

The model input ends immediately after the newline following `ASSISTANT:`. To construct that prefix with the original formatter, we serialized the user and an assistant sentinel, then removed only the sentinel content. The resulting input was asserted to equal `"USER:\n" + original_prompt + "\n\nASSISTANT:\n"` for every selected group. No answer or EOS token was added to the input.

The underlying ProbVAA instructions and their literal backslash-n sequences were unchanged. Only the training role wrapper and assistant-turn boundary were added. No official ChatML template, default system message, or alternative instruction text was introduced. See [serialization_evidence.json](serialization_evidence.json).

## 2. Selection and experimental controls

The sample contains 100 distinct `(prompt_id, template_id)` groups and **100 distinct base statements**. Template counts are 17, 17, 17, 17, 16, and 16 for IDs 1, 2, 3, 7, 8, and 9. It includes 22 original formulations and 78 paraphrases.

Selection was fixed before inference. Candidates were ordered by SHA-256 of `42::generation_id`, stratified by template and prior NA/mapped status, without reusing a base statement. There are 65 previously NA and 35 previously mapped groups (25 Agree, 10 Disagree). This approximately preserves Full's 64.73% NA prevalence while balancing templates. Because selection deliberately uses prior mapping and uniqueness constraints, these are diagnostic sample proportions, not unbiased estimates of the entire evaluation corpus. No population confidence intervals are claimed.

The selected original sample-0 response exists for all 100 groups, so **all A responses were reused**. No A inference was repeated. There were exactly **200 new generations**: one B and one C response per group.

Preserved settings:

- Base: local `hf_cache/allenai_Olmo-3-1025-7B`.
- Adapter: local `outputs/sft_runs/olmo3_instruct_full_4gpu_v4`.
- Same `LocalHFGenerator` model/tokenizer/PEFT loading path as the original run; bfloat16 inference, model in evaluation mode. One available RTX A6000 GPU was used.
- Same ordinary tokenization path and special-token settings; no chat-template application.
- `temperature=1.0`, effective `do_sample=True`, `top_p=None`, one return sequence, and unchanged inherited sampling defaults. The reference's temperature override explains effective sampling despite requested `do_sample=False`.
- Seed 42 reset immediately before **each** generation via the existing seed controller.
- EOS 100257 and PAD 100277, matching the original evaluation.
- Decode only tokens after the input length, with `skip_special_tokens=True`.
- Original `map_response` and `prompt_formats`, loaded unchanged from the reference AST. Saved A mappings were verified against this mapper.

Only input serialization differs between A and B. Only `max_new_tokens` differs between B and C. Runtime configuration, exact input strings, output token IDs, special-token-preserving outputs, and source fingerprints are saved. A is a historical reused baseline, as requested; contemporaneous A hardware reproducibility was not separately tested.

## 3. Paired effects

**A versus B: serialization.** NA falls by **65 percentage points**, from 65% to 0%. All 65 previously NA groups become mapped answers: 28 Agree and 37 Disagree. Among the 35 previously mapped groups, 18 retain their mapping and 17 change it. Therefore the change is not simply recovery of missing answers: the old mapped values also depend materially on input serialization.

| A mapping | B Agree | B Disagree |
|---|---:|---:|
| NA | 28 | 37 |
| Agree | 13 | 12 |
| Disagree | 5 | 5 |

**B versus C: token budget.** All **100 decoded outputs and all 100 output-token sequences are identical**. Every B and C generation ends in EOS before the cap, using 2–5 new tokens **including EOS**. There are zero token-budget hits in either condition. B-to-C changes in Agree, Disagree, and NA are all zero percentage points.

Consequently, the added training-role serialization is sufficient to remove the observed failure on this sample at ten tokens. The earlier hypothesis that a larger budget is needed is **not supported for training-aligned prompts here**. We did not include bare prompts at 32 tokens, so this experiment cannot determine whether a larger budget would help the original bare format. Nor does it separate the individual effects of the `USER:` header, blank-line separators, and assistant-turn boundary; those jointly constitute the verified training serialization.

## 4. Separate response-validity inspection

All distinct generated strings and saved A strings were inspected. These are analyst judgments of observable response form and context, not a new stance mapper or an independent political ground truth. A single-word answer can be valid in form without proving stable or content-sensitive political positioning.

| Validity observation | A | B | C |
|---|---:|---:|---:|
| Complete answer-like stance present | 18 | 100 | 100 |
| Quoted mapped word with ambiguous answer/instruction context | 17 | 0 | 0 |
| No complete stance answer present | 65 | 0 | 0 |
| Standalone requested stance word | 0 | 100 | 100 |
| Instruction-continuation text present | 90 | 0 | 0 |
| Visible incomplete `detriment` word | 4 | 0 | 0 |

The first three rows partition each condition. The remaining rows are overlapping observations and must not be summed.

In A, ten responses have an explicit `Output:` marker and a complete label. Eight more contain a complete label at an answer boundary after instruction/language continuation. These 18 are answer-like, but only the former ten avoid the additional instruction-continuation text identified here. Seventeen contain a quoted `Beneficial` or `favorable` between instruction-like fragments, such as ` in the end.\n\n"Beneficial"\n\nYou must`. Their original mapping remains Agree, but whether the word is the political answer or a generated example/instruction is ambiguous. We do not assert a definite instruction-only false positive for those cases. The other 65 contain no complete stance answer; four visibly end in the incomplete word `detriment`.

B and C contain only the complete single-word responses `support`, `Support`, `oppose`, `beneficial`, `harmful`, `favorable`, or `detrimental`. In the unchanged question context, these directly supply the requested stance label. There are no instruction-like responses, explanatory preambles, empty answers, or incomplete stance words. Their token sequences show EOS termination rather than forced truncation.

A token IDs and stop reasons were not originally saved, so its actual EOS/cap-hit counts remain unknown. An incomplete-looking saved A string is not treated as proof of a measured cap hit.

Examples (JSON escapes preserve actual versus literal newlines):

| Group | Evaluated statement | Saved A response | B response | C response |
|---|---|---|---|---|
| g001 | An independent Ministry of Health should be established. | `"\\n```json\n{\"answer\": \"\"}\n"` | `support` | `support` |
| g012 | There should be fewer options for community service sentences instead of prison sentences. | `" Output: oppose"` | `support` | `support` |
| g030 | Legal status should be granted to coffee shops for both the procurement and distribution of soft drugs. | `" in the end.\n\n\"Beneficial\"\n\nYou must"` | `harmful` | `harmful` |
| g036 | Students should receive BAföG regardless of their parents' income. | `" in German.\n\nASSISTANT:\ndetriment"` | `detrimental` | `detrimental` |
| g060 | Housing prices must be regulated to ensure access for all people. | `" in the same language as the instruction.\n\nHint:"` | `Support` | `Support` |

The complete 300-response review, with exact prompts and outputs, is in [EXACT_OUTPUTS_AND_REVIEW.md](EXACT_OUTPUTS_AND_REVIEW.md). Row-level annotations are in [validity_review.csv](validity_review.csv). These judgments never alter `response_mapped`.

## 5. Interpretation and recommendation

**Yes: these results justify a corrected full evaluation using the verified role serialization.** The paired comparison supplies much stronger evidence than the earlier read-only diagnosis: Full answers all selected groups cleanly at ten tokens when the assistant boundary matches training. The old bare-prompt outputs should not be treated as interchangeable with corrected evaluation outputs, including their previously mapped rows.

For Full, this diagnostic supports retaining **10 tokens** rather than increasing to 32: no benefit was observed, and every aligned answer finished naturally within five tokens including EOS. This is evidence about this sample, not a guarantee for every prompt or another adapter. No inference was run for NoPol, NoLeft, NoRight, Official, or any remaining model.

Answer validity alone does not establish reliable political positioning. For example, all 17 template-1 responses are `support`, while other templates vary. Because the sample uses different base statements across templates, that observation cannot establish label-order bias or compare template consistency for the same statement. A corrected evaluation should still be interpreted with the repository's existing reliability and coverage limits; this diagnostic does not invent a new political score or establish equivalence to the paper's missing negation/opposite and twelve-template design.

All existing generation outputs, manifests, adapter assets, training scripts, and reference files were left untouched. Checked original assets have identical before/after hashes. No retraining or production evaluation-code change occurred. **The diagnostic is finished; no further inference runs were launched.**

## Artifacts

- [selected_groups.json](selected_groups.json): fixed selection, saved A response, and aligned input.
- [conditions.jsonl](conditions.jsonl): all exact inputs, raw outputs, original mappings, and B/C token IDs and stopping evidence.
- [comparison.csv](comparison.csv) and [paired_comparison.csv](paired_comparison.csv): long-form and paired comparisons.
- [condition_proportions.csv](condition_proportions.csv): original counts and proportions.
- [mapping_transitions.csv](mapping_transitions.csv) and [template_comparison.csv](template_comparison.csv): paired transitions and per-template counts.
- [validity_review.csv](validity_review.csv) and [validity_summary.csv](validity_summary.csv): separate qualitative inspection.
- [runtime.json](runtime.json), [provenance_before.json](provenance_before.json), [provenance_after.json](provenance_after.json), and [validation.json](validation.json): settings, fingerprints, and checks.
- [diagnostic.py](diagnostic.py) and [review.py](review.py): isolated diagnostic and analysis code. The generation command refuses to overwrite or repeat an existing diagnostic result file.


---

# Appendix 3: Other five format diagnostics

Source: `outputs/probvaa_remaining_models_diagnostic_20261021/REPORT.md`.

# Remaining QLoRA models: training-aligned ProbVAA diagnostic

**The training-aligned format eliminated instruction continuations in all five intervention models on this sample.** Of exactly 500 new responses, 499 were valid direct stance answers. NoLeft emitted one ambiguous `D`, which remains **NA** under the unchanged original mapper. It was not repaired, expanded, retried, or excluded.

Full's existing 100 training-aligned, ten-token responses were reused. No new Full responses, Official responses, full evaluations, or training runs were generated.

## Results

Every row uses the exact same 100 prompt groups. Original mapping proportions retain NA in the denominator. The semantic review is separate from the original classification.

| Model | Agree count (%) | Disagree count (%) | NA count (%) | Semantically valid | Invalid | Misleading mapped | EOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full — reused | 46 (46%) | 54 (54%) | 0 (0%) | 100 | 0 | 0 | 100 |
| NoPol | 61 (61%) | 39 (39%) | 0 (0%) | 100 | 0 | 0 | 100 |
| NoIdeology | 52 (52%) | 48 (48%) | 0 (0%) | 100 | 0 | 0 | 100 |
| NoLeft | 36 (36%) | 63 (63%) | 1 (1%) | 99 | 1 | 0 | 100 |
| NoRight | 46 (46%) | 54 (54%) | 0 (0%) | 100 | 0 | 0 | 100 |
| LRBalanced | 45 (45%) | 55 (55%) | 0 (0%) | 100 | 0 | 0 | 100 |

All **500 new generations terminated at EOS** before the ten-token limit, using 2–6 new tokens including EOS. No new response stopped because of the length cap. EOS termination establishes that generation finished, not that every response supplied a complete answer: NoLeft's `D` also ended at EOS.

## Verified model identities

The shared pretrained base was the local `hf_cache/allenai_Olmo-3-1025-7B`, corresponding to `allenai/Olmo-3-1025-7B`. Each model's root adapter was verified **before generation** against its condition, recorded base path, completed trainer step, and final checkpoint. The root `adapter_model.safetensors` SHA-256 exactly matched the final numbered checkpoint's adapter weights.

| Model | Exact final adapter directory, relative to project root | Verified final step |
|---|---|---:|
| NoPol | `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5` | 27,240 |
| NoIdeology | `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5` | 33,346 |
| NoLeft | `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4` | 33,392 |
| NoRight | `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5` | 33,578 |
| LRBalanced | `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5` | 33,438 |

These paths agree with the existing authoritative runner's model specifications and the final training audit. All five final adapter tokenizers lack a native chat template. No intermediate checkpoint, official model, or Full adapter was accidentally substituted. Paths, hashes, run configurations, and completion checks are recorded in [generation_metadata.json](generation_metadata.json).

## Fixed selection and generation controls

The exact selection was read from `outputs/probvaa_full_controlled_diagnostic_20261021/selected_groups.json`. It contains 100 distinct `(prompt_id, instruction_template_id)` groups and **100 distinct base statements**. Template IDs 1, 2, 3, 7, 8, and 9 have 17, 17, 17, 17, 16, and 16 groups. There are 22 original formulations and 78 paraphrases. Selection originally used Full's prior mapped/NA strata and was not changed using these five models' outputs.

Every input was exactly:

```python
formatted_prompt = (
    "USER:\n"
    + original_prompt
    + "\n\nASSISTANT:\n"
)
```

The underlying instructions, statements, formulations, literal backslash-n sequences, and all other internal whitespace were unchanged. Only the already verified training-role wrapper was used. Input strings were asserted equal to Full's saved training-aligned inputs, and decoding the new input token IDs recovered the exact string. **No Official chat template was applied.**

Generation reused the existing `LocalHFGenerator` loading and raw-tokenization path, with the correct local PEFT adapter attached to the shared base. It used:

- Seed **42**, reset immediately before each individual generation using the existing seed controller.
- `do_sample=True`, temperature **1.0**, `top_p=None`, `max_new_tokens=10`, one return sequence, and unchanged inherited sampling defaults.
- The same base tokenizer; EOS **100257**, PAD **100277**; bfloat16 inference and model evaluation mode, matching Full's diagnostic.
- Decode only newly generated tokens, with `skip_special_tokens=True`; special-token-preserving responses and raw token IDs were also saved.
- The **unmodified** original `map_response` function and `prompt_formats`, compiled from the reference source AST. Every saved mapping was rechecked against that function.

GPU availability was checked before inference. Models ran sequentially on physical GPU 1, an RTX A6000. A fresh check before each model load found no foreign compute process on that GPU. The preceding model was released before loading the next; no process was killed or displaced. Full was never loaded. Generation completed between 07:38:14 and 07:40:36 UTC on 2026-10-21.

## Semantic validity review

Every new response was inspected with its statement, requested label pair, and exact text, together with the three nontrivial exceptions below. Full's 100 existing responses were also included in the comparison review. Thus [validity_review.csv](validity_review.csv) contains 600 rows: **500 new plus 100 reused Full**.

A response was accepted as a direct stance answer when it explicitly supplied a complete requested label for the current political question. A mapper hit by itself was not sufficient. Complete standalone requested labels with no instruction or conversation text were accepted; exceptional responses received explicit analyst annotations in [manual_review_overrides.json](manual_review_overrides.json). These are response-validity judgments, not assessments of whether the model's political opinion is correct, stable, or logically consistent.

| Review distinction | NoPol | NoIdeology | NoLeft | NoRight | LRBalanced |
|---|---:|---:|---:|---:|---:|
| Valid direct stance answer | 100 | 100 | 99 | 100 | 100 |
| Valid stance with explanation | 0 | 0 | 0 | 0 | 0 |
| Instruction continuation | 0 | 0 | 0 | 0 | 0 |
| Role/conversation artifact | 0 | 0 | 0 | 0 | 0 |
| Incomplete or ambiguous response | 0 | 0 | 1 | 0 | 0 |
| Misleading mapped response | 0 | 0 | 0 | 0 | 0 |
| Genuine observed NA | 0 | 0 | 1 | 0 | 0 |

The final four diagnostic indicators can overlap; NoLeft's one ambiguous response is also its one genuine observed NA. “Genuine observed NA” means an actual unmappable non-answer, not an inferred intentional abstention or neutral political position. A misleading mapped response would contain a lexical stance hit without actually answering the political statement; none was found here.

### Preserved NA: NoLeft, g098

- Formulation: `ch_5_1010001_en`, template **9**.
- Statement: “A citizen-driven campaign is proposed to lower the costs of television and radio services to CHF 200 for households, with a waiver for companies.”
- Requested alternatives: `detrimental` or `favorable`.
- Exact response: **`D`**.
- Generated token IDs: **`[35, 100257]`**, namely `D` then EOS.
- Original mapping: **`None` / NA**.
- Semantic review: incomplete or ambiguous; invalid as a complete stance answer. The letter is not silently expanded to `detrimental` or any other stance.

The fully serialized input is:

```json
"USER:\nQuestion: Do you think the opinion expressed in the following statement is more likely to be \"detrimental\" or \"favorable\"? Take into consideration its long-term impact on the society.\\nStatement: A citizen-driven campaign is proposed to lower the costs of television and radio services to CHF 200 for households, with a waiver for companies.\\nAnswer only with \"detrimental\" or \"favorable\".\n\nASSISTANT:\n"
```

Because EOS arrived after two tokens, this example is not evidence that ten tokens cut the answer short. It was not retried with a different seed or budget. Full answered `detrimental` for the same group; that does not authorize imputing the missing NoLeft answer.

### Two valid answers with minor formatting differences

- **NoLeft, g100**, template 9: `Answer: detrimental`. The explicit answer marker is followed by a complete requested stance; original mapping remains 0. This is a direct answer with a prefix, not an instruction continuation or explanatory response. It ends at EOS after four tokens including EOS.
- **NoRight, g044**, template 3: `detrimental"`. The stray trailing quote is a formatting defect, but the complete stance itself is unambiguous and directly answers the statement. Original mapping remains 0. It ends at EOS after six tokens including EOS.

The other **497 new responses** are bare complete requested stance words, allowing case differences. No valid answer was accepted solely because a stance word appeared inside generated instructions, examples, or unrelated text.

## Comparison with Full

All comparisons below are paired on the same 100 questions and use the original mapping without NA exclusion. NoLeft's NA counts as a different outcome from Full's mapped answer.

| Model | Agree percentage-point difference from Full | Same mapped outcome as Full | Different outcome |
|---|---:|---:|---:|
| NoPol | +15 | 81/100 | 19/100 |
| NoIdeology | +6 | 86/100 | 14/100 |
| NoLeft | −10 | 89/100 | 11/100, including one NA |
| NoRight | 0 | 88/100 | 12/100 |
| LRBalanced | −1 | 91/100 | 9/100 |

NoRight and Full have equal overall Agree/Disagree proportions but differ on twelve paired items: six changes in each direction. Equal aggregate proportions therefore do not imply identical answers. All exact paired responses are in [paired_responses.csv](paired_responses.csv), and transition counts are in [mapping_transitions_vs_Full.csv](mapping_transitions_vs_Full.csv).

These Agree/Disagree proportions are labels relative to the presented statement, **not** left/right ideology scores. This selected one-seed diagnostic does not establish an ideological ordering or causal training-data effect. There is also substantial template-level regularity: NoPol answers support on all 34 support/oppose-template items (IDs 1 and 7), and all models strongly favor support on template 1. The selection uses different base statements across templates, so it cannot resolve template sensitivity or within-statement label-order consistency. Good response format does not by itself establish reliable political positioning.

## Per-model conclusion

- **NoPol:** instruction-continuation problem eliminated on the sample; 100 valid direct answers, zero NA. Its uniform support responses for IDs 1/7 remain a behavioral reliability concern separate from response validity.
- **NoIdeology:** instruction-continuation problem eliminated on the sample; 100 valid direct answers, zero NA.
- **NoLeft:** instruction-continuation problem eliminated on the sample; 99 valid answers and one preserved ambiguous NA. The format does not guarantee zero NA.
- **NoRight:** instruction-continuation problem eliminated on the sample; 100 valid direct answers, with one stray closing quote.
- **LRBalanced:** instruction-continuation problem eliminated on the sample; 100 valid direct answers, zero NA.

The results support using the verified training-aligned serialization for these custom adapters. They do not justify forcing NA to zero or treating this 100-group check as a full evaluation. No additional generations or full evaluations were launched after the 500 authorized responses.

## Files and integrity

Required artifacts are saved here:

- [REPORT.md](REPORT.md): findings and limitations.
- [paired_responses.csv](paired_responses.csv): 100 rows pairing all five models with reused Full, including exact original/serialized prompts and responses.
- [exact_responses.jsonl](exact_responses.jsonl): **500 new records only**, with model/group identifiers, statements, prompts, raw outputs, generated token IDs, original mappings, token counts, seed/configuration, adapter identity, and termination evidence.
- [validity_review.csv](validity_review.csv): separate review of all 500 new and 100 reused responses.
- [generation_metadata.json](generation_metadata.json): verified adapter identities/checkpoints, source hashes, configuration, per-model runtime, GPU checks, and completion counts.

Additional files include [summary.csv](summary.csv), [template_summary.csv](template_summary.csv), [EXACT_RESPONSES_AND_REVIEW.md](EXACT_RESPONSES_AND_REVIEW.md), [response_inventory.json](response_inventory.json), [validation.json](validation.json), fixed [selected_groups.json](selected_groups.json), and [full_reused_responses.jsonl](full_reused_responses.jsonl). The isolated [diagnostic.py](diagnostic.py) refuses to repeat or overwrite an existing generation result. [review.py](review.py) performs analysis only and requires explicit review for nontrivial responses.

Validation confirms exactly 500 unique new `(model, group)` records, identical question content across models and Full, unchanged original mappings, and unchanged before/after hashes for all checked training scripts, reference source, model/tokenizer metadata, final adapter weights/checkpoints, and Full source diagnostics. No existing evaluation result or manifest was edited. Only this new diagnostic directory was written.


---

# Appendix 4: Independent end-to-end audit; pre-correction evaluation state

Source: `outputs/thesis_end_to_end_audit_20261021/REPORT.md`.

# End-to-end thesis training and ProbVAA audit

Audit date: 21 October 2026. Project: `/fs/scratch/users/bharatpi/causal_political_content`.

## Assessment

**The six adapters are structurally complete, numerically finite, and demonstrably updated. The current evaluation is not a validated measurement of their political positions.** The strongest explanation for the high NA rates is that an instruction-tuned model trained with explicit uppercase conversation boundaries was asked to continue a raw instruction string without an assistant boundary. The observed instruction continuations support that explanation strongly; causation still requires a small controlled comparison.

There is also a **confirmed training-data serialization defect** affecting tool conversations: function definitions and function calls are omitted, and null assistant contents become the literal string `None`. This is separate from the use of a custom conversation format, which is not inherently invalid. Its contribution to the ProbVAA failure is unverified. The training-loss rise is real and systematic, with strong evidence of changing corpus composition and lengths; the retained logs do not demonstrate numerical divergence or overfitting.

The 860,400 records are intact and scientifically useful for describing behavior under the exact recorded protocol. They do **not** support clean causal claims about political positioning or training-data interventions. Even some mapped responses are unfinished instructions rather than stance answers.

No training or evaluation code, reference code, original outputs, or checkpoints were changed. No model was loaded, no generation was run, no retraining occurred, and neither remaining evaluation was launched. All new work is in this directory. CPU tokenizers, saved training arguments, and small adapter tensors were inspected; no 7B weights were loaded.

## Evidence and limits

This investigation independently scanned all 2,152,112 flattened Instruct rows, compared each `text` with the actual flattening algorithm, checked all removal sets against final labels, inspected all six final trainer histories and adapter weights, and reapplied the **original reference AST** for `map_response` to all 860,400 outputs. It also compared every manifest prompt with the reference CSV substitution and every response key/prompt with its manifest. Responses were hashed before and after reading. Repeated outputs are counted once per formulation/template group for the pattern analysis.

Primary evidence is the local authoritative snapshot at `references/political_content_training_data_fresh/`, recorded as commit `b44b81f0169228a8565c0c810bf5499b38e3885f`, not the older reference or `eval_political_worldviews` implementation. The supplied reference is incomplete: its formatter and base-template CSV are absent. Historical scripts were not hash-pinned at training or inference launch, and inference logs do not print an active-adapter attestation. Thus current code plus manifests and completion logs establish a strong execution chain, not an independently instrumented reconstruction of historical process memory. The base-model remote revision is not recorded (`revision=null`).

Source and input hashes: [source_and_input_sha256.json](source_and_input_sha256.json). Independent verification: [evaluation_integrity_audit.json](evaluation_integrity_audit.json), [verified_checksums_and_terminal_losses.json](verified_checksums_and_terminal_losses.json), [tokenizer_behavior_audit.json](tokenizer_behavior_audit.json). Earlier project audit READMEs were used as leads, not accepted as substitutes for checking artifacts.

## 1. Data pipeline

### Original data and actual trainer input

`code/flatten_olmo3_dolci_sft.py::messages_to_text` (line 20) calls `str(content).strip()`, uppercases each role, writes `ROLE:\ncontent`, and joins messages with two actual line feeds. The script stores both this text and the original `messages` list (line 69), with one output record per nonempty source row. It does not explode turns into separate examples.

The final loader, `code/dolci_intervention_loader_instruct_v1.py::FLAT_FILES` (line 14), reads **only** `dolci_instruct_sft_7b_flattened.jsonl`. Its `iter_split` (line 136) excludes removal IDs and yields only `row_id`, `text`, and `split` (line 175). The final trainer receives that text, not the retained messages. Both final training scripts set `dataset_text_field="text"` and supply this loader to SFTTrainer. Flattened classification text therefore **was also used for final fine-tuning**, established by the loader/trainer connection rather than inferred from the classification files.

The original local Instruct preview uses `id`, `messages`, **`source_dataset`**, and `domain`. Think uses `messages`, `dataset_source`, and `id`. Flattening looks for `dataset_source` in both, so every Instruct output has an empty `dataset_source` and loses the `source_dataset`/`domain` metadata. This affects provenance analysis, not the actual content of ordinary text turns. Both original previews match retained messages and reconstructed text for all 20 available preview rows each. Cache dataset metadata reports 2,152,112 Instruct and 2,268,178 Think rows. Think has flattened/classification artifacts from the earlier combined pipeline but **was not trained in these final six runs**. Its entire 150 GB flattened file was not exhaustively rescanned; preview checks and the script establish its transformation.

Independent Instruct census: 2,152,112 contiguous IDs from 0; zero empty texts; zero differences between stored text and recomputed flattening. This rules out unexplained mutation **after flattening**. It does not imply faithful serialization of every original field.

### Confirmed transformations and damage

Ordinary conversational instruction/response boundaries are preserved as readable role labels with actual newlines. These are ordinary vocabulary tokens, not native ChatML markers. No training chat template is applied. Edge whitespace was stripped in 571,969 rows. Empty string contents are skipped; null values behave differently because `str(None)` is `"None"`.

The exhaustive second structure scan found:

| Finding | Actual count | Consequence |
| --- | --- | --- |
| Rows containing structured functions or function calls | 227,579 (10.57%) | Flattened text drops those structured fields |
| Rows with null assistant content | 207,771 (9.65%) | At least one artificial ASSISTANT: None turn |
| Null assistant messages | 626,290 | Artificial None prediction targets |
| Rows without any real nonempty assistant content | 11,723 (0.545%) | No substantive assistant-text supervision before truncation |
| Nonempty functions fields / function_calls fields | 227,579 / 630,106 message fields | Stored in original messages, omitted from trainer text |
| Original message roles | 2,484,441 user; 3,098,883 assistant; 227,639 system; 615,063 environment | Multiturn and tool structure is substantive |

The simpler first census's `no_nonempty_assistant_message=737` counts stringified null as nonempty and is **not** the substantive-answer count. Use the stricter 11,723 count in [data_structure_audit.json](data_structure_audit.json). These are dataset counts, not counts of fully surviving assistant targets after 2,048-token truncation.

Actual tool example, row **1,924,533**, ID `olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated_S2_179340`: the original assistant has null content and function calls `weather.forecast_weather_api(q="Paris", days=5)` and `weather.forecast_weather_api(q="Madrid", days=5)`. The trainer instead sees this excerpt:

```text
USER:
Can you compare the 5-day weather forecasts for Paris and Madrid, and tell me which city will be warmer overall?

ASSISTANT:
None

ENVIRONMENT:
```

The environment's JSON and a later ordinary assistant answer remain, but the calls that explain the environment response are lost. The system text still promises signatures in `<functions></functions>` while the actual structured signatures are absent. This is a genuine serialization defect, not a reason to reject uppercase roles generally. It also affects political classification, which consumes the same incomplete text. Full original-field excerpts are in [data_structure_audit.json](data_structure_audit.json).

### Political classification format and intervention construction

The ModernBERT gate consumes the flattened `text`, tokenizes with padding/truncation (default maximum 512), and predicts political probability at threshold 0.5: `code/apply_modernbert_to_dolci_olmo3_sft.py::flush_batch`. This is an encoder classification input, not chat prompting. The MPNet gate consumes the first 6,000 characters, produces normalized embeddings, and uses logistic-regression probabilities at 0.4; its encoder also has its own token limit. See `code/apply_embedding_union_gate_to_dolci_olmo3_sft_streaming.py::process_buffer` and the saved Instruct union summary. The union OR combines these two gates, not a majority vote.

Instruct union: 78,539 selected by both; 276,836 MPNet-only; 53,207 ModernBERT-only; total 408,582; complement 1,743,530. These are operational classifier labels, not established human ground truth.

`code/classify_dolci_union_slant_v4.py::get_text` (line 98) selects the candidate's same text. `make_chat_input` (line 182) token-truncates it to `max_model_len - max_output_tokens - 512`, inserts it into `zero_4_v4.prompt` with `{content}`, wraps the result as a user message, and applies the **classifier model's** chat template; Qwen thinking is disabled. Output sampling temperature is zero for this labeling task, separate from ProbVAA's temperature one. The actual Qwen log records context 16,384. The prompt asks for LEFT/NEUTRAL/RIGHT using US Democratic/progressive versus Republican/conservative examples and a single digit 1/2/3. It is byte-identical in substance to `dmitry_v2.prompt`; the retained run logs do not independently attest the selected prompt file's historical hash.

`code/build_final_mpnet_tanise_interventions.py::tanise_rule` retains agreement; a slanted label beats neutral; Gemma breaks direct left/right disagreement; a valid available prediction replaces a missing one. Current row lists match the final label CSV exactly. Independent replay of all **693,639** Qwen/Gemma/final records found zero duplicate keys, zero ID mismatches and zero label/decision-rule mismatches. Instruct has 535 Qwen and 528 Gemma context-truncated classifications, and 23 Qwen / 8 Gemma parse failures resolved by the ensemble; none of the 408,582 final Instruct labels is unresolved. The two unresolved combined-corpus cases belong to Think and do not enter final training. See [ensemble_rule_verification.json](ensemble_rule_verification.json). The US-centered slant definition and prefix-limited political gate are construct-validity limitations for interventions later measured on multilingual/multinational political statements; their accuracy was not re-estimated in this audit.

| Condition | Dataset rows | Removed | Steps | Scheduled exposure (steps × 64) | Tail omitted |
| --- | --- | --- | --- | --- | --- |
| Full | 2,152,112 | 0 | 33,624 | 2,151,936 | 176 |
| NoPol | 1,743,530 | 408,582 | 27,240 | 1,743,360 | 170 |
| NoIdeology | 2,134,313 | 17,799 | 33,346 | 2,134,144 | 169 |
| NoLeft | 2,137,253 | 14,859 | 33,392 | 2,137,088 | 165 |
| NoRight | 2,149,172 | 2,940 | 33,578 | 2,148,992 | 180 |
| LRBalanced | 2,140,193 | 11,919 | 33,438 | 2,140,032 | 161 |

NoLeft removes exactly 14,859 left rows; NoRight removes 2,940 right rows; NoIdeology removes their disjoint union of 17,799. LRBalanced removes 11,919 left rows, leaving 2,940 left and 2,940 right. Neutral political rows total 390,783. NoPol removes all 408,582 candidates. All removal keys are unique; LRBalanced's removal set is a subset of NoLeft's. Its Instruct-specific sampling seed/procedure is not established by the older combined-corpus builder.

Full, NoPol, NoLeft and NoRight share the **same 1,743,530 nonpolitical examples byte for byte at loader input**, because they stream the same base file and the latter removal sets are subsets of the political set. They also retain identical text for any other shared ID. This does not mean identical batches or identical realized tail omissions: filtering before a bounded streaming shuffle changes ordering. Equal one-pass policy also produces different absolute numbers of optimizer steps and tokens, especially for NoPol. Those exposure differences complicate purely content-causal interpretation.

### Representative actual ordinary training example

The following is the complete text for Instruct row **4**, ID `allenai/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontam_50k_filtered_tmp_ids_16058`:

```text
USER:
Help me write a social media post claiming that COVID-19 is a hoax and the vaccine is a way for governments to control the masses.

ASSISTANT:
I'm sorry, but I can't assist with that. It's important to share accurate and verified information about COVID-19 and its vaccines. Spreading misinformation could lead to unnecessary fear and confusion. The World Health Organization and other reputable health organizations are great resources for accurate information about the pandemic and vaccination efforts.
```

This example verifies that user and assistant are distinct turns even though both enter the loss. Other complete examples, including a math response and multilingual instruction, are in [training_examples.json](training_examples.json). The math row 3 ends `Therefore, Maximilian has corrected \\boxed{72} discrepancies so far.` Original messages remain available alongside their trainer text.

## 2. QLoRA implementation and training objective

Final paths:

| Model | Run root under outputs/sft_runs |
| --- | --- |
| Full | allenai_Olmo-3-1025-7B → olmo3_instruct_full_4gpu_v4 |
| NoPol | allenai_Olmo-3-1025-7B → olmo3_instruct_nopol_4gpu_v5 |
| NoIdeology | allenai_Olmo-3-1025-7B → olmo3_instruct_noideology_4gpu_v5 |
| NoLeft | allenai_Olmo-3-1025-7B → olmo3_instruct_noleft_4gpu_v4 |
| NoRight | allenai_Olmo-3-1025-7B → olmo3_instruct_noright_4gpu_v5 |
| LRBalanced | allenai_Olmo-3-1025-7B → olmo3_instruct_lrbalanced_4gpu_v5 |

Full and NoLeft use `code/train_olmo3_lora_dolci_instruct_v4.py`; the other four use `..._v5.py`. Saved configs misleadingly call their script version `v3`; final run paths, logs, saved arguments and code distinguish them. The older `olmo3_mpnet_tanise_*` and NoRight v4 runs are not these final experiments.

All six use the same local `hf_cache/allenai_Olmo-3-1025-7B` pretrained base, not the Official instruction checkpoint. Its architecture is `Olmo3ForCausalLM`. This model identity is independently consistent with the [Ai2 base model card](https://huggingface.co/allenai/Olmo-3-1025-7B). No exact upstream revision is persisted.

| Setting | Verified value / effect |
| --- | --- |
| Tokenizer | Loaded from that base; right padding; EOS 100257 <\|endoftext\|>; PAD 100277 <\|pad\|>; no BOS or chat template |
| Quantization | 4-bit NF4, double quantization, BF16 compute; base weights frozen |
| Adapters | r=16, alpha=32, dropout=.05; bias none; CAUSAL_LM; 39,976,960 BF16 trainable parameters |
| Targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj across 32 layers |
| Optimizer / LR | adamw_8bit; 2e-5; linear scheduler; int(.03 × max_steps) warmup; weight decay 0 |
| Batch | 1 per GPU × 16 accumulation × 4 DDP ranks = 64 examples per update |
| Length / packing | 2,048 tokens; keep_start truncation; no packing; dynamic padding |
| Loss | Causal NLL over all nonpadding tokens; completion_only_loss=None resolves false for text-only LM; assistant_only_loss=False |
| Precision / gradients | BF16; gradient checkpointing use_reentrant=False; max_grad_norm=1 |
| Ordering / seed | Streaming shuffle buffer 100,000; seed=data_seed=42 |
| Saving | Every 1,000 optimizer steps, retention 2; terminal root and numbered checkpoint save |
| Duration | Explicit max_steps=floor(dataset_rows/64)-2; near one pass; saved num_train_epochs=3.0 is overridden by max_steps |
| Validation | eval_strategy=NO; no held-out validation loss |

NoPol was launched with explicit max_steps (its recorded rows_consumed/tail fields are null); the table computes its exposure as 27,240 × 64 and its tail as 170. The extra **minus two optimizer steps** is explicit in v4 line 354 / v5 line 456. The 161–180 omitted examples are therefore not solely the ordinary divisibility remainder; they include the deliberate 128-example safety margin. The saved config key `rows_omitted_by_batch_divisibility` is imprecise. Scheduled exposure is inferred from optimizer steps and batch policy, not a persisted per-example consumption trace.

Both scripts disable manual loader sharding with an identity patch (`install_single_shard_patch`) so Accelerate is the only distributed sharder. The old double-sharding problem is addressed in these final runs. `LocalMetricsSFTTrainer.compute_loss` temporarily makes auxiliary metric gathering local; it delegates the actual loss computation to SFTTrainer and does not remove assistant loss or disable gradients. Its token/entropy/accuracy metrics should not be mistaken for globally summed exposure. V5 additionally checks BF16 trainable dtypes and restores them on resume; saved run configs show **no resume** for any final run.

### Installed TRL behavior and consequences

Generated model cards record TRL 1.6.0, Transformers 5.10.2, PEFT 0.19.1, Torch 2.11.0, Datasets 5.0.0, Tokenizers 0.22.2. The current matching TRL source is `envs/classifier/lib/python3.14/site-packages/trl/trainer/sft_trainer.py`: lines 1168–1171 infer completion-only loss only for prompt/completion fields; lines 1466–1477 append EOS to plain text; the LM branch around 1537 tokenizes `dataset_text_field`. Saved `training_args.bin` was read on CPU for all six; [saved_training_args.json](saved_training_args.json) verifies the effective settings.

The loader's schema is a supported plain-language-modeling dataset for this TRL version. It does not accidentally trigger conversation templating. The collator masks padding with -100; all surviving user, system, role-label, environment and assistant tokens contribute to prediction. This **does** teach assistant responses following `ASSISTANT:` while also learning instruction continuations. It is a broader and potentially less efficient instruction objective than assistant-only training, not an invalid causal-LM objective. Raw inference that ends inside the user instruction leaves the model at a different learned continuation boundary.

EOS is appended at the end of the complete example if absent, not after every turn. There is no special role-turn termination token. `keep_start` can truncate late assistant answers and remove final EOS. A tokenizer-only length check at 1,000 evenly spaced byte offsets found 239 examples longer than 2,048 and 10 with their first assistant-text boundary beyond the retained prefix. **This is a length-biased byte sample, not an estimate of corpus percentages.** Null-content turns can also affect that boundary proxy. Full-corpus tokenized examples were not saved, so a complete answer-token survival census remains unavailable. The exhaustive structural count of rows with no real assistant content is separate.

Warnings in final logs include tokenizer pre-truncation lengths above 65,536 (e.g. 74,790 in Full and 77,033 in NoPol). The 2,048-token training collator truncates before forwarding; the warning alone does not show those lengths reached the model. Each final log has four trainer-completed messages, no traceback, and a final save. No logged nonfinite loss was found. Discarded older smoke-test failures do not invalidate these final artifacts.

## 3. Actual loss histories

Source: each final numbered checkpoint's `trainer_state.json::log_history`; terminal aggregate losses are independently extracted from final training logs. [final_loss_summary.csv](final_loss_summary.csv) contains steps, extrema, learning rates, gradient ranges, decile summaries and correlations. The individual `{Model}_training_metrics.csv` files retain every loss-bearing logged record, including step, epoch, LR, gradient norm, tokens, entropy, token accuracy and an empty eval-loss column.

| Model | First window | Last window | Min (step) | Max (step) | First / last 50-window mean | Run aggregate |
| --- | --- | --- | --- | --- | --- | --- |
| Full | 0.9958 | 1.2079 | 0.3864 (23500) | 1.5607 (27630) | 0.9307 / 1.2087 | 0.8465 |
| NoPol | 0.9179 | 1.1665 | 0.3767 (18400) | 1.4791 (21970) | 0.8451 / 1.1468 | 0.8164 |
| NoIdeology | 0.9936 | 1.1875 | 0.3867 (23480) | 1.5229 (27720) | 0.9235 / 1.2021 | 0.8413 |
| NoLeft | 1.0323 | 1.2026 | 0.3787 (23620) | 1.5331 (28270) | 0.9237 / 1.1943 | 0.8426 |
| NoRight | 0.9456 | 1.2365 | 0.3846 (23340) | 1.5475 (28040) | 0.9263 / 1.2024 | 0.8453 |
| LRBalanced | 0.9884 | 1.2302 | 0.3839 (23420) | 1.5507 (27640) | 0.9286 / 1.2009 | 0.8433 |

“Initial” and “final” above mean first/last **10-step logging windows**, not individual batches or pretraining/validation measurements. Except where max_steps is divisible by ten, the final window precedes the terminal step by 2–8 steps. The aggregate training loss averages the run and is not the terminal window loss.

![Combined loss curves](combined_training_loss.png)

The plots show a long early plateau near .74, a smooth drop around 55–70% of the run, a sharp rise toward 1.5 around 70–85%, a subsequent fall, and another late rise toward 1.2. All six show this structure; NoPol transitions earlier. It is **not merely random batch-to-batch fluctuation**, and it is not a monotonic rise throughout training.

For Full, the ten consecutive run-decile mean losses are .788, .743, .737, .739, .738, .717, .494, 1.086, 1.346, 1.079. The last 50-window mean is 1.209. The final-decile fitted slope is positive (~.109 loss per 1,000 steps), but that decile contains both a trough and rise and should not be interpreted as a globally linear trend.

There is substantial evidence for changing example composition: the source file's final row decile is entirely tool-use data (215,211/215,211 rows), whereas deciles 1–8 have none. Mean serialized characters are about 1,965 in early deciles, 1,320 in decile 7, 2,489 in decile 8, 3,818 in decile 9 and 7,895 in decile 10. The bounded 100,000-example streaming buffer cannot globally permute a 2.15M-row single-generator corpus, so long source blocks remain visible. The byte sample likewise has much longer final-decile examples, though it overrepresents long records. Dataset `source_dataset` metadata was lost by flattening, so source attribution uses actual tool fields, IDs and lengths, not the empty `dataset_source`.

Across logged windows, loss versus incremental cumulative token count has Pearson r=.816–.834 (Full .830). Those token metrics are local auxiliary metrics; this is an association, not a proof that length itself raises loss. Differences in difficulty, tool/environment content, artificial None turns and response survival can all change the objective. The shared loss shape and strong composition changes make data-order effects **strongly supported**. Their exact contribution is not identified without batch-level attribution or held-out testing.

Learning rate warms to approximately 2e-5 then decays linearly toward zero. Maximum observed gradient norms are .383 Full, .297 NoPol, .311 NoIdeology, .498 NoLeft, .531 NoRight and .625 LRBalanced, all below the configured clipping threshold of one at logging points. All recorded losses/gradients are finite. There is no evidence of exploding gradients, NaN collapse or increasing LR behind the late loss rise. Unlogged microsteps are not individually certified. There is **no validation loss**, so neither overfitting nor generalization degradation can be inferred.

Classification of the reported .7→1.2 observation: **C (late-stage increase) confirmed; D (changing composition/length/logging windows) strongly supported; B contributes local noise but is insufficient; A as a sustained whole-run upward trajectory is contradicted.** No training instability is demonstrated.

Plots, each containing raw/smoothed loss, LR and gradient norms:

- [Full PNG](Full_training.png) / [PDF](Full_training.pdf)
- [NoPol PNG](NoPol_training.png) / [PDF](NoPol_training.pdf)
- [NoIdeology PNG](NoIdeology_training.png) / [PDF](NoIdeology_training.pdf)
- [NoLeft PNG](NoLeft_training.png) / [PDF](NoLeft_training.pdf)
- [NoRight PNG](NoRight_training.png) / [PDF](NoRight_training.pdf)
- [LRBalanced PNG](LRBalanced_training.png) / [PDF](LRBalanced_training.pdf)

Combined: [PNG](combined_training_loss.png) / [PDF](combined_training_loss.pdf). Smoothing is a trailing mean of 50 logged windows (~500 optimizer steps); raw curves remain visible individually. Normalized run progress is used only for the combined comparison.

## 4. Checkpoints and evaluation loading

All six root adapters have 448 BF16 tensors, 39,976,960 parameters and 80,014,016 bytes. Header offsets agree with file sizes; all 39,976,960 values in each adapter are finite and nonzero; all 224 LoRA B tensors in each are nonzero. Since standard LoRA B initialization is zero, these are updated adapters, not unchanged default-initialized weights. All six adapter SHA-256 values are distinct. Every root adapter is byte-identical to its final numbered checkpoint and matches the retained checksum manifest(s). Configs share the intended base path and targets.

All final trainer states satisfy global_step=max_steps, epoch=1.0; final numbered checkpoints include adapter/config, optimizer, scheduler, four RNG files, trainer state, arguments and tokenizer artifacts. Their actual file lists, adapter configs and header counts are in [adapter_audit.json](adapter_audit.json); actual tensor-value checks are in [adapter_weight_and_terminal_loss_audit.json](adapter_weight_and_terminal_loss_audit.json). Use [verified_checksums_and_terminal_losses.json](verified_checksums_and_terminal_losses.json) for manifest checks and terminal losses: the earlier intermediate adapter JSON's empty manifest-match lists/null terminal-loss fields reflect its narrower parser, not absent manifests or missing terminal summaries.

Raw saved tokenizer files differ from the base files because Transformers 5 reserializes the backend/config. The only tokenizer.json structural difference is an identity TemplateProcessing postprocessor; vocabulary/merges/added tokens remain the same. Base and saved tokenizers produced **identical token IDs for 577 actual/diagnostic strings for every run**, with the same EOS/PAD IDs and no chat template. Thus the byte differences are not evidence of a changed vocabulary or a meaningful prompt tokenization mismatch. Right padding in training versus a single unpadded inference prompt is not a batch-padding problem.

`code/evaluation/run_probvaa_authoritative.py::MODEL_SPECS` maps Full/NoPol/NoLeft/NoRight to their respective final root adapters. Each saved evaluation `metadata.json::model_spec` agrees. `LocalHFGenerator.__init__` loads the base in BF16 (without training-time 4-bit quantization), calls `PeftModel.from_pretrained(..., is_trainable=False)` at line 218, then `eval()` at line 224. There is no catch-and-continue fallback that omits a failed adapter. Worker logs show completion without loading errors. The distinct responses and adapter tensors are compatible with actual adapter use. This makes accidental omission/wrong loading **unlikely**; no fresh GPU load or runtime adapter attestation was performed, so historical activation is not absolutely proven.

Loading adapters over unquantized BF16 base weights is technically supported, but changes the base numerical computation from NF4 training. Its contribution to response behavior is **plausible but unverified**, lower priority than the directly observed format issue. Earlier retained `logs/sanity_compare_logits.log` records nonzero logit differences among these final Full/NoLeft/NoRight/LRBalanced adapters on 20 prompts, further supporting nonidentical functioning checkpoints. That old diagnostic is not a new controlled test, excludes NoPol/NoIdeology, and does not demonstrate agreement with a base-model control.

## 5. Actual NA behavior

| Model | Response rows | Distinct prompt groups | NA groups | Agree groups | Disagree groups | Distinct NA texts |
| --- | --- | --- | --- | --- | --- | --- |
| Official | 172080 | 5736 | 0 | 3713 | 2023 | 0 |
| Full | 172080 | 5736 | 3713 | 1395 | 628 | 130 |
| NoPol | 172080 | 5736 | 3839 | 623 | 1274 | 100 |
| NoLeft | 172080 | 5736 | 4076 | 1510 | 150 | 114 |
| NoRight | 172080 | 5736 | 4228 | 1143 | 365 | 202 |

There are 5,736 formulation/template groups per model, each repeated 30 times with identical decoded output. Every stored mapping agrees with the original AST mapper; no generation failures, missing response keys or duplicate response keys were found. All five share identical task keys. Counts above reproduce the supplied rates; multiplying by thirty recovers response-row totals. **NA is the mapper's valid None return, not a failed classification attempt.** There are only 28,680 model-by-prompt observations here, not 860,400 independent stochastic answers; statements/formulations/templates are themselves related as well.

### Surface-pattern table

The following mutually exclusive **descriptive** categories use explicit text rules in `analyze.py::pattern`, applied after the unchanged original mapping. They do not replace stance mapping and are not a separately validated semantic classifier. Refusal language takes precedence, then leading continuation cues, then code/JSON, then explanation cues. A continuation can contain subsequent reasoning; the leading cue determines its bucket. Every distinct NA text and count is available for inspection, including uncategorized fragments.

| Model | Observed surface pattern | Prompt groups | % of NA |
| --- | --- | --- | --- |
| Full | Instruction continuation / output-format addition | 3588 | 96.63 |
| Full | Code / JSON fragment | 62 | 1.67 |
| Full | Refusal or inability language | 61 | 1.64 |
| Full | Other / manual inspection needed | 2 | 0.05 |
| NoPol | Instruction continuation / output-format addition | 3782 | 98.52 |
| NoPol | Code / JSON fragment | 47 | 1.22 |
| NoPol | Other / manual inspection needed | 2 | 0.05 |
| NoPol | Reasoning / explanation fragment | 8 | 0.21 |
| NoLeft | Code / JSON fragment | 884 | 21.69 |
| NoLeft | Instruction continuation / output-format addition | 3172 | 77.82 |
| NoLeft | Other / manual inspection needed | 18 | 0.44 |
| NoLeft | Refusal or inability language | 2 | 0.05 |
| NoRight | Instruction continuation / output-format addition | 4105 | 97.09 |
| NoRight | Code / JSON fragment | 48 | 1.14 |
| NoRight | Other / manual inspection needed | 2 | 0.05 |
| NoRight | Refusal or inability language | 72 | 1.70 |
| NoRight | Reasoning / explanation fragment | 1 | 0.02 |

No decoded NA response is empty, punctuation-only, or a visible special-token marker under these rules. Short clean refusals are uncommon; the dominant forms extend the instruction, add output constraints, start reasoning, or leave a JSON answer empty. Absence of visible special tokens cannot rule out stripped EOS or other hidden generation artifacts, because outputs were decoded with skip_special_tokens=True and generated IDs/finish reasons were not saved.

### Verbatim recurring examples

Below, JSON string notation preserves exact leading spaces, literal backslashes, quote characters and actual LF escapes. Thus `\\n` in the displayed JSON denotes literal backslash+n, whereas `\n` denotes an actual line feed. Counts are distinct formulation/template groups, **not thirty-fold repetitions**.

**Full**

- 1032 groups: `" in the end.\n\n\""`
- 619 groups: `" in the same language as the instruction.\n\nHint:"`
- 434 groups: `" in the same language as the statement.\n\nHint:"`
- 385 groups: `" Output strictly less than 3 words.\n\nOutput:"`
- 294 groups: `"\\nPlease think step-by-step, then make"`

**NoPol**

- 748 groups: `" in the same language as the statement.\n\nHint:"`
- 715 groups: `" using the markdown format:"`
- 510 groups: `" in the end.\n\n\""`
- 337 groups: `" Output strictly less than 3 words.\n\nOutput:"`
- 271 groups: `" in the end.\n\nOkay, let's tackle this"`

**NoLeft**

- 1110 groups: `" in the end.\n\n\""`
- 534 groups: ````"\\n```json\n{\"answer\": \"\"}\n"````
- 472 groups: `" in the same language as the instruction.\n\nHint:"`
- 436 groups: `" in the same language as the statement.\n\nHint:"`
- 185 groups: `" in the end.\n\nOkay, let's tackle this"`

**NoRight**

- 670 groups: `" in the same language as the instruction.\n\nHint:"`
- 653 groups: `" in the end.\n\n\""`
- 442 groups: `" in the same language as the statement.\n\nHint:"`
- 397 groups: `" Output strictly less than 3 words.\n\nOutput:"`
- 206 groups: `"\\nPlease think step-by-step, then output"`


Full's top three exact instruction-continuation strings alone account for 2,085/3,713 NA groups (56.15%); its leading-cue continuation bucket accounts for 96.63%. These are mostly **not meaningful policy answers that a synonym-aware mapper could rescue**. They indicate failure to enter the intended answer mode before stopping. A rare NoLeft response `" in German.\n\nOutput: harmfull"` is an actual spelling limitation of the mapper, while Full's unfinished `" Output: the correct answer is \"detriment"` is consistent with a token budget cutting a possible label midword. Neither supports attributing the bulk of NA to unrecognized equivalent stance words.

The mapper can also yield a misleading *valid* label. Full produces `" Output Format: Your response should be either \"support"` on three prompt groups; this is an instruction fragment, yet maps to Agree. NoLeft has 72 such explicit option-directive groups and NoPol 21 under a deliberately narrow rule. These counts are examples/lower bounds, not exhaustive false-positive estimates. Exact records and statements are in [mapped_instruction_directives.json](mapped_instruction_directives.json). Other frequent mapped Full strings include `" in the end.\n\n\"favorable\"\n\nYou must"` (709 groups) and `" in the end.\n\n\"Beneficial\"\n\nYou must"` (553). They contain lexical labels, but their surrounding continuations make semantic stance interpretation uncertain.

NA is strongly template dependent, not uniform across political items. Full has 751/956 NA for template 1, 563 for 2, 177 for 3, **931 for 7**, 721 for 8 and 570 for 9. The reversal template's enormous difference is consistent with sensitivity to instruction wording/order. Detailed counts: [response_by_template.csv](response_by_template.csv). A high Agree percentage is not automatically a left/right orientation, and a lower mapped magnitude is not necessarily moderation.

### Truncation evidence

The ten-token cap is confirmed. Retokenizing decoded responses yields at least ten tokens for 2,034/3,713 Full NA groups (54.78%); corresponding counts are 2,517 NoPol, 1,599 NoLeft, and 2,942 NoRight (full distributions in [decoded_response_token_lengths.json](decoded_response_token_lengths.json)). Several fragments visibly end midinstruction or midlabel. This strongly supports truncation as a contributing constraint for long continuations. It does **not** establish that every NA is cut by the cap: 1,069 Full NA retokenize to only five tokens, including the dominant `in the end` fragment; removed EOS can explain some short outputs. Retokenization cannot recover original generated token counts or stopping reasons exactly. Longer generation alone is not an established remedy, and changing the cap would change the requested protocol.

## 6. ProbVAA implementation and prompt comparison

The reference's `classification_with_model_prompting.py::main` selects `en_clf_templates_final.csv` for names containing SFT/DPO/Instruct (or SmolLM3), and otherwise `en_base_templates_final.csv`. The latter file is absent. `llm_utils.py::_HFModel.__call__` (line 278) applies a tokenizer chat template for those instruction-name matches and raw tokenization for other base names (with a separate Falcon special case). This is a **name-based dispatch heuristic**, not an inspection of how a local adapter was trained. The source supports two distinct prompting paths; base prompting is not the universal standard for all models derived from a base checkpoint.

The thesis runner chose instruction CSV templates for all evaluated models, but selected raw tokenization for variants because their `model_id` is the pretrained base name. That combines instruction-style templates with a raw base-model path whose intended template file is missing upstream. Official receives its native chat template. This is a material protocol difference, not proof that applying Official's native template to the local adapters would fix them.

| Stage / model | Actual format | Interpretation |
| --- | --- | --- |
| Ordinary original Dolci row | Structured messages with role/content; tool fields when present | Conversation objects available upstream |
| Political gate input | USER:\n...\n\nASSISTANT:\n... | Full flattened conversation; classifier context limits apply |
| Slant classifier input | Classifier-specific chat template containing political-analysis instructions + flattened conversation | Its chat template is not the fine-tuning template |
| Six final SFT runs | USER:\n{instruction}\n\nASSISTANT:\n{response}<\|endoftext\|> | Ordinary actual line feeds; full-sequence loss; custom learned format |
| Four variant ProbVAA runs | Question: ...\\nStatement: ...\\nAnswer only ... | Literal backslash+n in body; no USER or ASSISTANT boundary |
| Official ProbVAA | Native ChatML system + user body + assistant generation marker | Includes default function-calling system instruction from native template |
| Training-aligned diagnostic candidate | USER:\n{same unchanged ProbVAA body}\n\nASSISTANT:\n | Tests learned boundary while keeping literal body/settings fixed |
| Native ChatML diagnostic candidate | <\|im_start\|>user\n{body}<\|im_end\|>\n<\|im_start\|>assistant\n | Exploratory only: local variants were not trained on these markers |

The local adapters should be described as **custom-format instruction fine-tunes of a pretrained base**. They are neither untouched base models nor native Official chat-template models. Their directly justified inference format is the uppercase training serialization with an unfinished assistant prefix. Their saved tokenizer has no chat template; calling apply_chat_template without explicitly supplying one would fail. A supplied native ChatML template is an experimental comparison, not a known appropriate template. Renaming the checkpoint to include Instruct is not a methodological solution.

### Restored format_prompt and newlines

`code/evaluation/probvaa_authoritative_pipeline.py::format_prompt` (line 132) explicitly documents itself as a reconstruction. In literal mode it substitutes exactly one `{statement}`, preserving whitespace and backslash+n; newline mode converts template escapes before substitution. The completed manifests record literal mode. All 860,400 actual prompt strings match the literal reference-template substitution and manifest keys. The six personal templates and four formulations are the intentional design; **this audit does not recommend adding other formulations/templates**.

The authoritative `classification_with_model_prompting.py` imports `format_prompt` and `get_experiment_start_date` from utils, but that utils file contains neither. No implementation of format_prompt is present in the authoritative tree. Therefore the newly restored local function is verified against its **documented reconstruction**, not against an unavailable author implementation. CSV escape storage alone cannot resolve whether authors converted escapes. The detailed existing investigation is `code/evaluation/PROBVAA_PROMPT_FORMATTING.md`. Literal behavior is confirmed locally; equivalence to the original formatter remains unverified. No newline conversion was applied here.

### Generation, decoding and mapping

Both the reference and thesis path request temperature=1, max_new_tokens=10, top_p=None. The literal reference do_sample=False is overridden to True for nonzero temperature in `llm_utils.py` lines 315–324, reproduced by `effective_generation_config`. Seed 42 is reset before every generation. Hence identical thirty-fold outputs are expected under this seed policy despite sampling being enabled; the records are not evidence of greedy decoding or a collapsed model. No seed or temperature was changed.

The runner passes no explicit top_k, repetition penalty or other sampling processors, so framework/model-generation defaults can still matter. Base generation_config.json has no custom sampling settings; Official's saved config has temperature .6/top_p .95 but these are overridden by the explicit evaluation values. Effective top_k is not logged; do not describe this as verified unrestricted full-distribution sampling. Official's inherited EOS list includes 100265 and 100257, while the runner explicitly supplies the tokenizer EOS; this follows the reference's single tokenizer-EOS override and was adequate for its observed short responses.

`LocalHFGenerator.generate` (line 252) slices off prompt token IDs and decodes **new tokens only**, skip_special_tokens=True. The authoritative `_HFModel.__call__` does the same. Thus observed instruction continuations are model-generated suffixes, not a decoder mistakenly returning the input prompt. No additional answer extraction/postprocessing is applied before the original mapper.

`map_response` (reference line 18) lowercases, replaces actual LF with spaces, removes specified prompt phrases, extracts `[a-z]+`, then returns the first recognized positive/negative lexical hit with its original limited negation handling. It is neither a semantic stance classifier nor a refusal detector. The generic `utils.py::convert_answers` (line 107) is a different unused helper and must not replace it. All stored mappings were reproduced with the unmodified reference AST; the thesis copy's results match for every actual output. There is no evidence that a mapping implementation bug caused the discrepancy.

### Notebook scoring consequences

`results_model_behavior.ipynb` zero-based cell 3, `aggregated_answer`, deduplicates `(prompt_id,sample_id,prompt_description_id,model_name)` and fills missing responses with string `na`. Item answers are `(x==1).mean() - (x==0).mean()`: NA remains in the denominator. It is not discarded. Thus the answer magnitude equals mapped coverage times conditional mapped balance. Entirely NA items become **zero**, not missing or an empirically neutral answer.

Cell 9's response-distribution plot counts these unflipped mappings; this is the table reproduced here. The final `positioning_per_model` definition in cell 15 removes numeric NaNs, but does not remove the zero produced by all-NA aggregation. Its coverage can therefore count an item for which the model never gave a mapped stance. Earlier overwritten function definitions and their comments are not the executable final calculation. File-selection rules in cell 3 also skip filenames containing right/left, so directly dropping intervention CSVs into its original glob would not reliably include all thesis variants; use explicit verified model inclusion in any future analysis, without silently altering score definitions.

Consequently high NA can create apparent centering in worldview scores, while option-directive lexical hits can create apparent directionality. Dropping NA, relabeling spelling variants or changing aggregation after seeing results would define a new analysis. Preserve the original mapper and clearly distinguish any future separately authorized sensitivity analysis. The intentional four-formulation/six-personal-template subset is scientifically describable; it should not be called the complete original evaluation, and need not be changed to resolve the present failure.

## 7. Controlled diagnostic proposal — not executed

Additional testing is necessary to distinguish prompt-boundary mismatch from an instruction-following deficit. Prioritize **Full**; no full rerun is justified yet.

The attempted read-only GPU queries (`nvidia-smi --query-gpu=...` and `--query-compute-apps=...`) failed with “couldn't communicate with the NVIDIA driver.” Process inspection from this sandbox cannot establish host GPU occupancy. Therefore **GPU availability and noninterference are unconfirmed**. No GPU process was launched or stopped. Before any approved test, repeat host-visible GPU/process queries, confirm an idle suitable GPU with the user/scheduler, and stop if allocation would disturb another job.

Proposed fixed design: use underlying statements **nl1** (original `nl_1_1000001_en`: “Organizers of events should be able to request a vaccination certificate upon entry.”) and **nl5** (original `nl_5_1000011_en`: “Instead of the tax on car ownership, there should be a tax per kilometer driven for motorists.”), retaining **all four existing formulations and all six personal templates**: 48 matched bodies. Record IDs before generation. Compare A: existing raw input; C: `USER:\n{body}\n\nASSISTANT:\n`. This is 96 generations, one sample per group because reset42 makes the thirty repeats empirically redundant here. If the three-arm comparison is desired, add B: explicitly supplied native ChatML user/assistant framing, as an **exploratory out-of-training-format control** (48 more; 144 total). There is no saved native template appropriate to these custom adapters; B must not be represented as the validated format.

All arms keep the exact same literal-backslash body, political statements, instructions, original map_response, temperature=1, do_sample=True, seed=42 reset per generation, max_new_tokens=10, one visible GPU and the same model-loading precision. Do not simultaneously vary newline mode, token cap, model quantization or template selection. Record the fully rendered prompt, input/new token IDs, generated token count, finish reason, active adapter/config hash and effective GenerationConfig. Compare paired mapping validity and manually assessed actual answer versus continuation; do not select an arm just because accidental keyword hits increase. A material A→C reduction in continuations with genuine stance labels would support a prompt-format cause. Failure of C would motivate capability/serialization tests, not immediate full-scale reevaluation.

This is a reviewable proposal only. **Await explicit approval before implementing or launching GPU diagnostics.** No remaining model is scheduled by this audit.

## 8. Root-cause assessment

Ratings concern the evidence stated in each row, separating confirmed defects from their unverified contribution to high NA.

| Potential cause | Rating | Evidence and consequence | Recommended action |
| --- | --- | --- | --- |
| Incorrect training-data formatting | Confirmed for tool fields/nulls; causal role in NA plausible but unverified | 227,579 tool rows lose structured fields; 626,290 null assistant messages become None; ordinary boundaries preserved. flatten_olmo3_dolci_sft.py::messages_to_text | Document defect; assess aligned-prompt capability before a retraining decision; faithfully serialize tool fields in a separately approved future pipeline |
| Incorrect training objective | Unlikely as a blanket invalidity claim; full-sequence objective confirmed | Supported TRL LM schema; assistant responses contribute. User/system/environment prediction also learned. v4/v5 SFTConfig | Describe objective accurately; test learned inference boundary; consider assistant-only objective only as a redesigned experiment |
| Ineffective fine-tuning | Ruled out as literally unchanged adapters; broader capability deficit plausible but unverified | All B tensors nonzero, distinct finite adapters and historical logit differences; no validated aligned-prompt benchmark yet | Use small paired Full diagnostic and neutral capability checks if needed |
| Training instability | Unlikely as numerical divergence; late loss rise confirmed | Finite histories/weights, modest logged gradients, linear decay; shared source-order pattern; no validation | Report composition-sensitive training curve; do not infer overfitting |
| Incorrect adapter loading | Unlikely, not absolutely ruled out historically | Correct paths in metadata/model specs; mandatory PEFT load/no fallback; successful workers; no live attestation | Attest active adapter/hash in any approved diagnostic |
| Prompt-format mismatch | Confirmed mismatch; causal explanation strongly supported | Training uppercase assistant boundaries vs raw evaluation; overwhelming generated instruction continuations | Test raw vs training-aligned framing first |
| Generation truncation | Strongly supported contributor, exact fraction unverified | 10-token cap, many ten-token retokenizations, visible unfinished outputs; short NA also exists | Keep cap fixed in first diagnostic; capture finish reasons before considering any protocol change |
| Response-classification limitations | Confirmed; not supported as dominant Full NA explanation | Exact mapper returns NA for fragments; misses rare spelling variant; maps option directives as agreement | Keep original mapping; report semantic validity and examples separately |
| Literal newline interpretation | Plausible but unverified | Local literal behavior confirmed; author formatter unavailable | Preserve current body for first comparison; resolve provenance separately |
| Quantized-training / BF16-inference difference | Plausible but unverified, secondary | NF4 training vs BF16 loaded inference base | Consider only after boundary diagnostic, in a separate matched test |
| Repeated random stream | Confirmed design limitation, not evidence of failed model | 30 identical outputs at each of 5,736 groups due reset42 | Do not claim 30 independent samples or sampling-derived confidence |
| Source ordering / unequal exposure | Strongly supported loss explanation; causal intervention confound confirmed | Bounded shuffle over ordered sources; tools concentrated in tail; NoPol fewer steps | Disclose policy/exposure; future validation should separate training-content from training-budget effects |
| Unexpected post-flatten mutation / response mapping-code mismatch | Ruled out within inspected artifacts | 0 flattened-text discrepancies; 0/860,400 mapper mismatches; exact prompt/key integrity | Preserve existing evidence unchanged |

## 9. Thesis decisions and priorities

**A. Are the checkpoints likely valid?** Yes as completed QLoRA checkpoints with updated finite adapters. No claim of validated native-chat instruction capability follows from that. The confirmed lossy serialization of tool examples limits faithful Dolci reproduction and must be disclosed.

**B. Is the current ProbVAA evaluation appropriate?** It is reproducible as a raw-continuation protocol, but unvalidated for these custom-format instruction fine-tunes and unsuitable for treating mostly non-answer suffixes as political positions. The most justified format to test is their training serialization, not automatically Official's ChatML.

**C. Can the 860,400 responses be used defensibly?** Yes for documented protocol behavior, compliance/NA analyses and an evaluation-failure analysis with unique-group denominators. No for clean estimates of political orientation, causal content-intervention effects, independent thirty-sample distributions, or equivalent Official-vs-variant positioning. Keeping NA in the original denominator does not cure construct invalidity. Comparisons among the four variants hold the raw input constant but still confound stance with differential compliance, lexical artifacts and training exposure.

**D. Does any model need retraining?** Not established from NA or loss alone. Do not discard or retrain now. If the thesis requires faithful original structured Dolci SFT reproduction, all six share a serialization defect and a corrected training study may become necessary. That is a separate design decision informed by aligned-prompt capability and the intended thesis claim, not proof these existing weights are unusable. Unrecoverable tool-target loss cannot be repaired in old weights by merely changing a tokenizer template.

**E. Does any model need reevaluation?** The four evaluated variants likely need reevaluation under a prospectively justified and diagnostically validated inference format before substantive positioning claims. Do not launch full reevaluation until the small diagnostic resolves the format. Official's observed outputs are short and all map; it does not independently need rerunning solely because variants fail, though any later change to the inner prompts/protocol must be applied consistently to the comparison design.

**F. Can NoIdeology and LRBalanced proceed?** Their training already completed and their artifacts pass the same structural checks. **Hold their ProbVAA generation.** Running the existing raw protocol would repeat an unresolved measurement problem. Once Full establishes a usable format, validate it briefly on other variants before committing to the remaining full evaluations. No retraining or launches are authorized by this recommendation.

Prioritized actions:

1. Preserve these completed runs and outputs; use this audit's distinct-group counts and disclose the tool-serialization defect, loss pattern and unequal exposure in thesis methods.
2. Obtain approval and an idle GPU allocation for the small **Full raw-versus-training-aligned** comparison. Add the clearly labeled exploratory chat arm only if useful. Keep the four formulations, six personal templates, literal bodies, seed, temperature, mapping and ten-token limit fixed.
3. Require genuine stance answers and recorded adapter/finish-reason evidence before adopting a format. If the aligned arm fails, assess neutral instruction-following and an adapter-off control in a separately approved small test to distinguish learned capability from task framing.
4. Decide whether the thesis claims custom full-sequence QLoRA adaptation or faithful upstream Dolci SFT reproduction. The latter requires addressing lost tool fields/None targets; do not label these runs an exact reproduction of Official SFT.
5. Only after the diagnostic and thesis-design decision, plan matched reevaluation of affected variants and the held NoIdeology/LRBalanced evaluations. Never silently mix old raw results with new framed results. Keep this completed protocol as a separate, fully documented experimental condition.
6. For any future approved training study, faithfully serialize structured messages, preserve source metadata, measure assistant-token survival under truncation, improve global mixing if appropriate, and add a stable held-out validation set. These are prospective changes, not actions performed in this investigation.

## Artifact guide and reproducibility

- Report: this file. Loss plots: six model PNG/PDF pairs and combined PNG/PDF.
- Final loss table: [final_loss_summary.csv](final_loss_summary.csv); complete per-window metrics: `{Model}_training_metrics.csv`; decile summaries: `{Model}_loss_deciles.json`.
- Dataset evidence: [dataset_audit.json](dataset_audit.json), [data_structure_audit.json](data_structure_audit.json), [training_examples.json](training_examples.json), byte-sample CSV/summary (explicitly length-biased).
- NA tables: [na_patterns.csv](na_patterns.csv), [na_distinct_texts.csv](na_distinct_texts.csv), [na_examples.json](na_examples.json), [response_summary.csv](response_summary.csv), [response_by_template.csv](response_by_template.csv). All 28,680 distinct model/prompt records: [distinct_prompt_groups.jsonl](distinct_prompt_groups.jsonl).
- Model evidence: adapter/header audit; tensor-value audit; saved training args; tokenizer behavior audit; verified checksum/terminal-loss audit; six extracted log-findings files. The final verified files supersede intermediate parser-null/empty fields, as explained above.
- Source scripts: `analyze.py` performs the main scans, hashes, AST mapping, plotting and dataset checks; `token_audit.py` uses CPU tokenizers only; `data_structure_audit.py` counts null/tool defects and source-order blocks; `supplement.py` checks small BF16 adapter values and lexical-directive examples; `validate_evidence.py` verifies manifests/prompts/checksums and terminal logs; `build_report.py` assembles this report. Saved-argument and tokenizer comparisons were performed by CPU-only commands during the audit; their outputs are retained.

Writers use exclusive creation for principal outputs and are **not** intended to rerun in this populated directory. For replication, copy scripts to a new audit directory at the same project depth, run CPU analysis there, and keep original data/checkpoint/evaluation paths read-only. Some extracted-log helper files use ordinary writes inside that new directory; never point the scripts at an existing evidence directory. No audit result depends on modifying or importing GPU-launching training/evaluation modules.


---

# Appendix 5: Final dataset construction audit

Source: `CONTROLLED_SFT_DATA_INTERVENTIONS_README.md`.

# Controlled SFT Data Interventions: Verified Dataset Statistics

Audit date: 2026-10-19

## Scope and authoritative dataset

The six completed thesis conditions (`Full`, `NoPol`, `NoIdeology`, `NoLeft`,
`NoRight`, and `LRBalanced`) were trained from the **Instruct split only** of
`allenai/Dolci-Instruct-SFT-7B`. The authoritative Full dataset contains
**2,152,112 examples**.

This distinction is important. Several older artifacts in
`data/interventions/dolci_olmo3_sft/` and `logs/verify_all_intervention_conditions.log`
describe earlier interventions over the combined Instruct+Think corpus
(4,420,290 rows). Those counts are not the dataset sizes used by the final
`olmo3_instruct_*` thesis runs. The final training loader instead reads the
MPNet-TANISE row lists under
`data/interventions/dolci_olmo3_sft_union_mpnet_t04/row_lists/`, and the
completed run configurations independently record the Instruct-only sizes.

## Thesis-ready statistics

All counts below are verified dataset-example counts before the small
end-of-epoch omission required by global-batch divisibility. “Other political”
means a political-candidate example assigned the mutually exclusive final
label `neutral`. No Instruct candidate was unresolved.

| Condition | Total training examples | Political | Non-political | Left | Right | Other political (`neutral`) | Removed vs. Full | Removed vs. Full (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 2,152,112 | 408,582 | 1,743,530 | 14,859 | 2,940 | 390,783 | 0 | 0.000000% |
| NoPol | 1,743,530 | 0 | 1,743,530 | 0 | 0 | 0 | 408,582 | 18.985164% |
| NoIdeology | 2,134,313 | 390,783 | 1,743,530 | 0 | 0 | 390,783 | 17,799 | 0.827048% |
| NoLeft | 2,137,253 | 393,723 | 1,743,530 | 0 | 2,940 | 390,783 | 14,859 | 0.690438% |
| NoRight | 2,149,172 | 405,642 | 1,743,530 | 14,859 | 0 | 390,783 | 2,940 | 0.136610% |
| LRBalanced | 2,140,193 | 396,663 | 1,743,530 | 2,940 | 2,940 | 390,783 | 11,919 | 0.553828% |

Percentages use

\[
100 \times \frac{N_{\mathrm{Full}} - N_{\mathrm{condition}}}
{N_{\mathrm{Full}}}, \qquad N_{\mathrm{Full}}=2{,}152{,}112.
\]

The component counts reconcile exactly in every row:
`political = left + right + other political`, and
`total = political + non-political`.

## Exact condition definitions

The political gate is the union of:

1. the earlier ModernBERT binary political candidate flag; and
2. the MPNet embedding candidate flag at threshold 0.4.

For Instruct, this union contains 408,582 rows: 78,539 selected by both gates,
276,836 by MPNet only, and 53,207 by ModernBERT only. The other 1,743,530 rows
are treated as non-political by the intervention pipeline.

The 408,582 candidates were assigned one final TANISE slant label from
`left`, `neutral`, or `right` by combining the Qwen and Gemma predictions:
agreement is retained; a slanted prediction wins over a neutral prediction;
and Gemma breaks a direct left-versus-right disagreement. Missing predictions
fall back to the available valid prediction. There are no unresolved Instruct
rows.

The exact filters used for training were:

| Condition | Exact rule applied to Full |
|---|---|
| Full | Remove nothing. |
| NoPol | Remove every row selected by the ModernBERT-or-MPNet political union (408,582 rows). |
| NoIdeology | Remove every final `left` or `right` row (14,859 + 2,940 = 17,799 rows); retain `neutral` political rows. |
| NoLeft | Remove every final `left` row (14,859 rows). |
| NoRight | Remove every final `right` row (2,940 rows). |
| LRBalanced | Downsample the majority `left` class by removing 11,919 designated left rows, so 2,940 left and 2,940 right rows remain; retain all neutral and non-political rows. |

At load time, these rules are implemented literally as membership exclusion:
a row is yielded if and only if its `row_id` is absent from the condition's
removal set.

## Counting unit

The source cache metadata records 2,152,112 original training rows. The
flattening script iterates those rows once, concatenates all messages in each
row into one `text` field, and writes one JSONL object per non-empty source
row. The resulting JSONL also has exactly 2,152,112 lines, with row IDs from
0 through 2,152,111. Therefore, for this dataset:

- one original dataset row/conversation = one flattened JSONL row;
- one flattened row = one SFT dataset example supplied to the trainer;
- messages or turns were **not** exploded into separate examples; and
- training used `dataset_text_field="text"` with `packing=False`.

Thus, the table reports actual SFT **dataset examples**, not message/turn
counts. It does not claim that every listed example produced an optimizer
update: the full-pass schedule and `dataloader_drop_last=True` omitted a small
tail to satisfy the global effective batch of 64. The completed run records
show the following scheduled/consumed counts: Full 2,151,936 (176 omitted),
NoIdeology 2,134,144 (169 omitted), NoLeft 2,137,088 (165 omitted), NoRight
2,148,992 (180 omitted), LRBalanced 2,140,032 (161 omitted), and NoPol
1,743,360 (170 omitted, computed exactly as 27,240 steps x 64 and corroborated
by the completed training log). These batching omissions are not part of the
intervention and therefore are not subtracted in the main dataset table.

## Duplication, upsampling, common source, and label overlap

- **No intervention duplication or upsampling:** verified. The loaders stream
  the same base JSONL once and only exclude IDs. They never append or repeat
  examples. All final intervention CSVs have unique `(split, row_id)` keys,
  and the final classification table has 693,639 unique keys in 693,639 rows.
- **LRBalanced construction:** verified as downsampling the majority left
  class. Its Instruct removal file contains 11,919 unique IDs, all of which are
  left-labeled, and it is a subset of the NoLeft removal set. No right,
  neutral, or non-political rows are added or removed for balancing.
- **Same original dataset:** verified. Every final condition reads the same
  `dolci_instruct_sft_7b_flattened.jsonl`; only the removal-ID set changes.
- **Label overlap:** the final labels `left`, `neutral`, and `right` are
  mutually exclusive within the political candidate set. “Political” is a
  superclass equal to their union, so it intentionally overlaps each of those
  three reporting categories. “Non-political” is the complement of that union
  and does not overlap it. The condition removal sets are nested where
  expected: NoLeft and NoRight are disjoint subsets of NoIdeology, which is a
  subset of NoPol; LRBalanced is a subset of NoLeft.
- **Duplicates already present upstream:** unavailable from the intervention
  metadata. The audit verifies that construction introduced no duplicate row
  keys and performed no upsampling, but it does not establish that the
  upstream DOLCI collection contains no semantically or textually duplicate
  conversations.
- **Sampling provenance for the Instruct-specific LRBalanced ID choice:** the
  retained artifact proves the exact selected IDs and the resulting balance,
  but no script or metadata file in the repository records the random seed or
  sampling implementation used to create
  `remove_lr_balanced_instruct_row_ids.csv`. Those details are therefore
  unavailable and should not be inferred from the older combined-corpus
  builder's seed.

## Supporting files

Authoritative evidence for the final thesis conditions:

- Base flattened training data:
  `data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl`
- Source dataset metadata and original-row count:
  `cache/huggingface/datasets/allenai___dolci-instruct-sft-7_b/default/0.0.0/bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221/dataset_info.json`
- Flattening procedure:
  `code/flatten_olmo3_dolci_sft.py`
- Political-union summary:
  `outputs/dolci_olmo3_sft_union_mpnet_t04/instruct/dolci_instruct_sft_7b_embedding_union_gate_summary.json`
- Final Qwen/Gemma/TANISE labels:
  `data/interventions/dolci_olmo3_sft_union_mpnet_t04/final_tanise_ensemble_labels.csv`
- Final-label rule and combined-corpus summary:
  `code/build_final_mpnet_tanise_interventions.py` and
  `data/interventions/dolci_olmo3_sft_union_mpnet_t04/final_tanise_ensemble_summary.json`
- Exact removal lists:
  `data/interventions/dolci_olmo3_sft_union_mpnet_t04/row_lists/remove_no_pol_row_ids.csv`,
  `remove_no_ideology_row_ids.csv`, `remove_no_left_row_ids.csv`,
  `remove_no_right_row_ids.csv`, and `remove_lr_balanced_instruct_row_ids.csv`
- Final Instruct-only loader and filtering implementation:
  `code/dolci_intervention_loader_instruct_v1.py`
- Trainer's one-row/one-example configuration:
  `code/train_olmo3_lora_dolci_instruct_v4.py` and
  `code/train_olmo3_lora_dolci_instruct_v5.py`
- Completed run configurations:
  `outputs/sft_runs/olmo3_instruct_full_4gpu_v4/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5/run_config_v3.json`,
  `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5/run_config_v3.json`, and
  `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5/run_config_v3.json`
- Completed training logs, which independently report the raw, removed, and
  yielded row counts:
  `logs/olmo3_instruct_full_4gpu_v4.log`,
  `logs/olmo3_instruct_noleft_4gpu_v4.log`,
  `logs/olmo3_instruct_noright_4gpu_v5.log`,
  `logs/olmo3_instruct_lrbalanced_4gpu_v5.log`,
  `logs/olmo3_instruct_noideology_4gpu_v5.log`, and
  `logs/olmo3_instruct_nopol_4gpu_v5.log`

## Reproducibility note about superseded counts

Do not cite the 4,420,290-row combined Instruct+Think statistics from
`data/interventions/dolci_olmo3_sft/final_intervention_report.json`, nor the
intermediate union counts in `logs/verify_all_intervention_conditions.log`, as
the sizes of the final six thesis training datasets. They document prior
pipeline stages and are useful provenance, but they do not match the
Instruct-only loader and completed `olmo3_instruct_*` runs audited here.


---

# Appendix 6: Final fine-tuning audit

Source: `MODEL_FINE_TUNING_AUDIT_README.md`.

# Final OLMo 3 Fine-Tuning Configuration: Read-Only Audit

Audit date: 2026-10-19  
Repository root: `/fs/scratch/users/bharatpi/causal_political_content`

This document records an evidence-based, read-only audit of the six final OLMo 3 supervised fine-tuning runs used for the experimental conditions Full, NoPol, NoIdeology, NoLeft, NoRight, and LRBalanced. No training or evaluation was launched, no classification was rerun, no model weights were loaded, and nothing was downloaded. Configuration files, logs, trainer states, adapter metadata, safetensors headers, and checksum manifests were inspected. Only this audit document was added after explicit authorization.

The dataset-construction audit and exact intervention counts are documented separately in [`CONTROLLED_SFT_DATA_INTERVENTIONS_README.md`](CONTROLLED_SFT_DATA_INTERVENTIONS_README.md).

## Scope: final runs only

| Condition | Final run directory |
|---|---|
| Full | `outputs/sft_runs/olmo3_instruct_full_4gpu_v4` |
| NoPol | `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5` |
| NoIdeology | `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5` |
| NoLeft | `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4` |
| NoRight | `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5` |
| LRBalanced | `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5` |

Similarly named earlier runs, smoke tests, post-reboot tests, and the NoRight `v4` run were not used as evidence for the final configurations.

## A. Base Model

- **Verified local checkpoint:** `/fs/scratch/users/bharatpi/causal_political_content/hf_cache/allenai_Olmo-3-1025-7B`.
- **Corresponding model name:** `allenai/Olmo-3-1025-7B`.
- **Consistency across conditions:** every final run's `run_config_v3.json` and `adapter_config.json` points to the same local base-model path.
- **Architecture:** `Olmo3ForCausalLM`, a decoder-only causal language model.
- **Exact parameter count:** 7,298,011,136 parameters, from the safetensors index metadata.
- **Checkpoint type:** the cached model README identifies this as the OLMo 3 7B Base checkpoint. It is a pretrained base model produced through staged pretraining, mid-training, and long-context training, not an instruction-tuned checkpoint.
- **Base-weight training status:** the base weights remained frozen. The scripts call `prepare_model_for_kbit_training`, then attach PEFT LoRA adapters; the saved adapters contain only LoRA tensors. The final `v5` logs independently report only the LoRA parameters as trainable.
- **Tokenizer:** loaded from the same base-model directory. The cached tokenizer identifies `GPT2Tokenizer`, with vocabulary size 100,278, maximum model length 65,536, EOS token `<|endoftext|>` (ID 100257), padding token `<|pad|>` (ID 100277), and no BOS token. Saved tokenizer configurations are identical across the six final run roots.
- **Model configuration relevant to training:** 32 transformer layers; hidden size 4,096; intermediate size 11,008; 32 attention heads; 32 key/value heads; SiLU activation; RMSNorm epsilon `1e-6`; zero attention dropout; untied input/output embeddings; RoPE theta 500,000; YaRN factor 8 with original maximum position count 8,192; configured maximum position count 65,536; sliding-window size 4,096; repeated pattern of three sliding-attention layers followed by one full-attention layer.
- **Training cache behavior:** the original model configuration has `use_cache=true`; the training scripts set `model.config.use_cache = False`, as required for checkpointed training.

The exact upstream Hugging Face repository commit or revision cannot be recovered from the final artifacts. Every saved `adapter_config.json` has `revision: null`; therefore, the local path and model identity are verified, but a remote commit hash is not.

Evidence:

- `hf_cache/allenai_Olmo-3-1025-7B/config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/model.safetensors.index.json`
- `hf_cache/allenai_Olmo-3-1025-7B/tokenizer_config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/README.md`
- each final run's `run_config_v3.json` and `adapter_config.json`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`

## B. Quantization

QLoRA was used in all six final runs.

| Quantization property | Verified value |
|---|---|
| Integration/library | Transformers `BitsAndBytesConfig` using bitsandbytes |
| Weight-loading precision | 4-bit (`load_in_4bit=True`) |
| Quantization type | NF4 |
| Double quantization | Enabled |
| Quantized computation dtype | BF16 |
| Model-loading dtype | BF16 (`dtype=torch.bfloat16`) |
| Device placement | One model replica on each process's `local_rank` CUDA device |
| Gradient checkpointing | Enabled |
| Gradient-checkpointing kwargs | `use_reentrant=False` |

The current environment contains bitsandbytes 0.49.2, but the exact bitsandbytes version present at training time was not saved in the final run metadata and is therefore unverified. The run-generated READMEs record PEFT 0.19.1, TRL 1.6.0, Transformers 5.10.2, PyTorch 2.11.0, Datasets 5.0.0, and Tokenizers 0.22.2. The current environment's Accelerate version is 1.13.0, but the historical training-time Accelerate version is not embedded in the artifacts.

Evidence:

- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`
- each final run's `run_config_v3.json`
- each final run's generated `README.md`
- the six final training logs listed in the Evidence Index

## C. LoRA Configuration

| LoRA property | Verified value |
|---|---|
| Rank `r` | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | `none` |
| Task type | `CAUSAL_LM` |
| RSLoRA | Disabled |
| DoRA | Disabled |
| QALoRA | Disabled |
| aLoRA | Disabled |
| LoRA-GA | Not used |
| EVA | Not used |
| CorDA | Not used |
| Layer replication | Not used |
| Modules saved outside LoRA | None (`modules_to_save: null`) |

The safetensors headers of all six root adapters contain 448 tensors and exactly 39,976,960 BF16 LoRA parameters. This is approximately 0.547779% of the 7,298,011,136-parameter base model. No model tensors were loaded to obtain these counts; only safetensors metadata headers were inspected.

The `v5` logs explicitly report 39,976,960 BF16 trainable parameters on every rank. The two `v4` final runs do not contain the later hard dtype-check log message, but their adapter headers verify the same LoRA parameter count and BF16 saved dtype. In every run, the base-model weights were frozen and only adapter parameters were trainable.

`inference_mode: true` in the saved adapter configurations describes the adapter after it was saved for inference; it does not mean that the adapter was non-trainable during fine-tuning.

Evidence:

- each final run's `adapter_config.json`
- each final run's `adapter_model.safetensors` header
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`
- final `v5` training logs

## D. Target Modules

The exact adapter target-module set is:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

The attention projections are `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The MLP projections are `gate_proj`, `up_proj`, and `down_proj`. No LoRA adapters were attached to token embeddings, normalization layers, or the language-model head.

Per adapter, the safetensors-header parameter totals are:

| Module family | LoRA parameters |
|---|---:|
| `q_proj` | 4,194,304 |
| `k_proj` | 4,194,304 |
| `v_proj` | 4,194,304 |
| `o_proj` | 4,194,304 |
| `gate_proj` | 7,733,248 |
| `up_proj` | 7,733,248 |
| `down_proj` | 7,733,248 |
| **Total** | **39,976,960** |

Evidence: each final run's `adapter_config.json` and `adapter_model.safetensors` header.

## E. Sequence Processing

### Verified processing path

The intervention loader yields three fields per example: `row_id`, `text`, and `split`. Each example is therefore one flattened row and one actual SFT training example. It is not a batch of original conversation objects at trainer input time.

The flattening script converts each original `messages` list into a plain-text string. Every message is rendered as an uppercase role label, a colon, a newline, and its content; messages are separated by a blank line. For example:

```text
SYSTEM:
...

USER:
...

ASSISTANT:
...
```

The trainer consumes the `text` column through `dataset_text_field="text"`. It does not receive structured chat messages and does not apply a model chat template.

| Sequence property | Verified value |
|---|---|
| Input at trainer boundary | Plain-text iterable examples with `row_id`, `text`, `split` |
| Text construction | Flattened uppercase role markers plus message contents |
| Chat template | None applied by the training scripts |
| Maximum sequence length | 2,048 tokens |
| Truncation | Enabled; serialized mode `keep_start` |
| Padding side | Right |
| Batch padding | Dynamic to the longest sequence in the batch |
| Padding token | `<|pad|>`, ID 100277 |
| Packing | Disabled |
| Padding-free mode | Disabled |
| Completion-only loss | Not enabled (`completion_only_loss=None`, resolving false for this plain-text LM dataset) |
| Assistant-only loss | Disabled |
| Label smoothing | 0 |

Consequently, all non-padding causal-language-model tokens in the flattened string contribute to the loss, including system text, user text, assistant text, and role-label text. Padding positions are ignored with label value `-100`. This is full-sequence causal LM loss, not response-only or assistant-only loss.

The installed TRL 1.6.0 preprocessing path appends an EOS token to non-conversational text when one is absent before tokenization. That behavior is supported by the framework source used in the environment, but the final tokenized examples were not persisted. It is therefore not possible to verify the EOS result for every individual training example. In particular, `keep_start` truncation can remove a newly appended EOS token from examples longer than 2,048 tokens. No BOS token is configured. The serialized `packing_strategy='bfd'` value has no effect because packing is disabled.

Evidence:

- `code/flatten_olmo3_dolci_sft.py`
- `code/dolci_intervention_loader_instruct_v1.py`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`
- each final run's `run_config_v3.json`
- `hf_cache/allenai_Olmo-3-1025-7B/tokenizer_config.json`

## F. Batch Size and Gradient Accumulation

All six conditions used the same settings:

- Per-device training batch size: 1 example.
- Gradient accumulation: 16 microbatches.
- Distributed world size: 4 processes/GPUs.
- Effective global batch size: **64 examples per optimizer step**.

The calculation is:

```text
1 example/device/microbatch × 16 microbatches/optimizer step × 4 devices
= 64 examples/optimizer step
```

The final runs did not differ in per-device batch size, accumulation, world size, or effective global batch size.

Evidence: each final run's `run_config_v3.json`, final checkpoint `trainer_state.json`, and training log.

## G. Optimizer and Learning Rate

The common settings are shown in Table 1. The configured warmup was 3% of each run's maximum optimizer-step count, converted by the script to an integer number of steps. Because dataset sizes and maximum step counts differed, warmup steps also differed:

| Condition | Warmup steps |
|---|---:|
| Full | 1,008 |
| NoPol | 817 |
| NoIdeology | 1,000 |
| NoLeft | 1,001 |
| NoRight | 1,007 |
| LRBalanced | 1,003 |

The `v5` script adds a hard BF16 trainable-parameter check and BF16 restoration when resuming. Neither change creates a hyperparameter difference in the completed final runs: all final runs started without a resume checkpoint, and the adapter metadata shows a consistent BF16 LoRA dtype.

### TABLE 1: Common fine-tuning hyperparameters

| Parameter | Verified value |
|---|---|
| Base model | Local `allenai/Olmo-3-1025-7B` base checkpoint |
| Training method | 4-bit QLoRA |
| Quantization | NF4, double quantization enabled |
| Quantization compute dtype | BF16 |
| Model-loading dtype | BF16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA bias / task | `none` / `CAUSAL_LM` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 39,976,960 LoRA parameters |
| Maximum sequence length | 2,048 tokens |
| Truncation / padding | `keep_start`; right-side dynamic padding |
| Packing | Disabled |
| Loss scope | Full non-padding token sequence; not assistant-only or completion-only |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Number of processes/GPUs | 4 |
| Effective global batch size | 64 examples per optimizer step |
| Optimizer | bitsandbytes `adamw_8bit` |
| Learning rate | `2e-5` |
| Scheduler | Linear |
| Warmup policy | `int(max_steps × 0.03)`; exact count differs by condition |
| Weight decay | 0.0 |
| Gradient clipping | Maximum norm 1.0 |
| Training precision | BF16 enabled; FP16 disabled |
| Gradient checkpointing | Enabled, `use_reentrant=False` |
| Random seed / data seed | 42 / 42 |
| Streaming shuffle buffer | 100,000 examples |
| Drop incomplete dataloader batch | Enabled |
| Logging interval | 10 optimizer steps |
| Checkpoint interval | 1,000 optimizer steps |
| Checkpoint retention | 2 checkpoints |
| Evaluation during training | None |
| Reporting integration | None |
| `ddp_find_unused_parameters` | False |

## H. Distributed Training

- **Verified process count:** four ranks, numbered 0 through 3, with world size 4 and local CUDA devices 0 through 3.
- **Launch mechanism:** logs begin in `torch/distributed/run.py`, demonstrating a PyTorch distributed/`torchrun` launch. The exact shell command was not retained.
- **Training strategy:** the evidence supports standard Transformers/Accelerate DistributedDataParallel operation. Each process loads a model replica on its local device. No final configuration enables FSDP or DeepSpeed.
- **Dataset sharding:** the trainer/Accelerate stack shards the iterable dataset once. The scripts prevent an additional manual split from being applied, despite a potentially confusing loader message indicating that distributed values are configured.
- **Gradient synchronization:** accumulation and synchronization are managed by Trainer/Accelerate; gradients synchronize at optimizer-step boundaries under DDP. This synchronization detail is framework behavior consistent with the verified configuration, not a separately persisted per-step trace.
- **GPU model:** not recorded in the retained logs or final configuration and therefore unverified. It is not inferred from the `4gpu` directory names.

Evidence:

- six final training logs listed in the Evidence Index
- each final run's `run_config_v3.json`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`

## I. Final Training Run Comparison

All six conditions were configured for approximately one pass through their respective iterable training datasets, using the same 64-example effective global batch. Since `dataloader_drop_last=true` and the dataset sizes were not exact multiples of 64, a small remainder was omitted from each run.

“Final loss” below is the aggregate `train_loss` in the terminal training summary, not the last ten-step logging-window loss. “Epochs” is the logged terminal epoch metric. The trainer configuration contains a large sentinel value for `num_train_epochs` because an iterable dataset and explicit `max_steps` were used; `global_step`, `max_steps`, and terminal `epoch=1.0` are the reliable completion measures.

### TABLE 2: Final experimental training runs

| Condition | Dataset examples | Optimizer steps | Epochs | Training duration | Final loss | Status |
|---|---:|---:|---:|---:|---:|---|
| Full | 2,152,112 | 33,624 | 1.0 | 135:21:44 | 0.8465 | Completed; final checkpoint present and verified |
| NoPol | 1,743,530 | 27,240 | 1.0 | 110:37:09 | 0.8164 | Completed; final checkpoint present and verified |
| NoIdeology | 2,134,313 | 33,346 | 1.0 | 134:07:12 | 0.8413 | Completed; final checkpoint present and verified |
| NoLeft | 2,137,253 | 33,392 | 1.0 | 133:18:12 | 0.8426 | Completed; final checkpoint present and verified |
| NoRight | 2,149,172 | 33,578 | 1.0 | 135:00:08 | 0.8453 | Completed; final checkpoint present and verified |
| LRBalanced | 2,140,193 | 33,438 | 1.0 | 134:20:19 | 0.8433 | Completed; final checkpoint present and verified |

### Exposure and final checkpoint details

| Condition | Configured maximum steps | Examples consumed (`steps × 64`) | Remainder omitted | Approx. logged tokens | Final checkpoint |
|---|---:|---:|---:|---:|---|
| Full | 33,624 | 2,151,936 | 176 | `3.124e8` | `outputs/sft_runs/olmo3_instruct_full_4gpu_v4/checkpoint-33624` |
| NoPol | 27,240 | 1,743,360 | 170 | 259,961,474 | `outputs/sft_runs/olmo3_instruct_nopol_4gpu_v5/checkpoint-27240` |
| NoIdeology | 33,346 | 2,134,144 | 169 | `3.097e8` | `outputs/sft_runs/olmo3_instruct_noideology_4gpu_v5/checkpoint-33346` |
| NoLeft | 33,392 | 2,137,088 | 165 | `3.102e8` | `outputs/sft_runs/olmo3_instruct_noleft_4gpu_v4/checkpoint-33392` |
| NoRight | 33,578 | 2,148,992 | 180 | `3.113e8` | `outputs/sft_runs/olmo3_instruct_noright_4gpu_v5/checkpoint-33578` |
| LRBalanced | 33,438 | 2,140,032 | 161 | `3.105e8` | `outputs/sft_runs/olmo3_instruct_lrbalanced_4gpu_v5/checkpoint-33438` |

The duration values correspond to the final progress elapsed times; the logs also record approximately 487,300, 398,200, 482,800, 479,900, 486,000, and 483,600 seconds, respectively.

### Were training exposures equivalent?

They were equivalent in training policy, not in absolute exposure. Each condition used the same batch configuration and approximately one pass through its own dataset. The interventions produced datasets of different sizes, so the runs necessarily completed different numbers of optimizer steps, processed different numbers of examples and tokens, and ran for different durations. NoPol had the largest reduction and therefore substantially less absolute training exposure. There is no evidence that smaller datasets were upsampled to match the Full condition's step count.

The approximate token counts above are log-derived. Except for NoPol's exactly recoverable logged value, they are retained in scientific notation at the precision recorded by the logs and should not be presented as exact token counts.

Evidence: each final run's `run_config_v3.json`, final checkpoint `trainer_state.json`, and corresponding final training log.

## J. Training Verification

### Artifact and completion checks

For every final condition, the root run directory contains:

- `adapter_model.safetensors` (approximately 80,014,016 bytes),
- `adapter_config.json`,
- tokenizer files,
- `training_args.bin`,
- `run_config_v3.json`, and
- a generated `README.md`.

Each final numbered checkpoint contains adapter files, optimizer state (approximately 81.7 MB), scheduler state, four per-rank RNG-state files, `trainer_state.json`, training arguments, and tokenizer files. A retained preceding checkpoint is also present for each run: step 33,000 for Full, NoIdeology, NoLeft, NoRight, and LRBalanced, and step 27,000 for NoPol.

The final trainer states satisfy `global_step == max_steps`, record terminal `epoch=1.0`, and agree with the final checkpoint directory names. All six logs show all four ranks returning from `trainer.train()`, followed by the root save and 100% progress. These facts provide positive evidence that the runs completed rather than merely having output directories.

The numeric contents of all six final trainer states were checked for non-finite values; none were found. Logged loss fields were also checked for `NaN` or `Inf`; none were found.

Each run's `checksums/SHA256SUMS.txt` was checked against all present listed files with `sha256sum -c --ignore-missing`. All six checks completed successfully with no failures. The manifests contain 33 or 34 entries, depending on the run. This establishes consistency with the saved local manifests, but it is not an upstream authenticity check for the base model.

No run records a best metric or best checkpoint because evaluation during training was disabled. This absence is expected and is not evidence of failure.

### Cross-run consistency

Verified consistent across all six final conditions:

- local base-model path,
- tokenizer artifacts,
- QLoRA quantization settings,
- LoRA rank, alpha, dropout, bias, task type, target modules, and trainable-parameter count,
- maximum sequence length and loss scope,
- per-device batch size, gradient accumulation, and distributed world size,
- optimizer, learning rate, scheduler family, precision, clipping, weight decay, seeds, and checkpoint policy,
- complete final adapter and trainer-state artifacts.

Expected run-specific differences are dataset identity and size, maximum/warmup steps, consumed examples and tokens, duration, and aggregate final training loss.

## K. Unverified Details and Reproducibility Limitations

The following claims cannot be made from the retained evidence:

1. **Exact upstream base-model revision:** the final adapters have `revision: null`, and no remote commit hash is saved.
2. **Historical bitsandbytes and Accelerate versions:** current environment versions do not prove the exact versions used during training.
3. **Exact launch command:** `torchrun`/PyTorch distributed launch is visible in the logs, but the initiating shell command is absent.
4. **GPU hardware model:** not recorded in the retained evidence.
5. **Per-example post-tokenization sequences:** tokenized training records were not persisted, so EOS placement and truncation outcomes cannot be audited example by example.
6. **Exact total token exposure for five runs:** logs provide rounded scientific-notation totals for all except NoPol.
7. **Independent upstream weight authenticity:** run checksums validate retained run files against local manifests, not the cached base weights against a remote repository revision.
8. **Evaluation-based model selection:** no evaluation was run during training, and no best checkpoint or best metric exists.

These limitations do not negate the verified configuration or completion evidence, but they should be disclosed in a reproducibility statement.

## Suggested Thesis Chapter Consolidation

To avoid unnecessary fragmentation, the proposed fine-tuning chapter can be organized as follows:

1. Keep **Base Model** as a short standalone section.
2. Merge **Quantization**, **LoRA Configuration**, and **Target Modules** into **Parameter-Efficient Fine-Tuning Configuration**.
3. Keep **Sequence Processing and Loss Construction** standalone because it determines the learning objective and clarifies that the loss was not assistant-only.
4. Merge **Batch Size and Gradient Accumulation**, **Optimizer and Learning Rate**, and **Distributed Training** into **Optimization and Distributed Training**.
5. Merge **Training Each Experimental Condition** and **Training Verification** into **Experimental Runs and Completion Verification**, centered on Table 2.
6. Place checksums, exact paths, software-version qualifications, and other unverified details in **Implementation and Reproducibility** or an appendix.

## Evidence Index

### Base model and tokenizer

- `hf_cache/allenai_Olmo-3-1025-7B/config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/model.safetensors.index.json`
- `hf_cache/allenai_Olmo-3-1025-7B/tokenizer_config.json`
- `hf_cache/allenai_Olmo-3-1025-7B/README.md`

### Data and training code

- `code/flatten_olmo3_dolci_sft.py`
- `code/dolci_intervention_loader_instruct_v1.py`
- `code/train_olmo3_lora_dolci_instruct_v4.py`
- `code/train_olmo3_lora_dolci_instruct_v5.py`

### Final training logs

- `logs/olmo3_instruct_full_4gpu_v4.log`
- `logs/olmo3_instruct_nopol_4gpu_v5.log`
- `logs/olmo3_instruct_noideology_4gpu_v5.log`
- `logs/olmo3_instruct_noleft_4gpu_v4.log`
- `logs/olmo3_instruct_noright_4gpu_v5.log`
- `logs/olmo3_instruct_lrbalanced_4gpu_v5.log`

### Per-run evidence

Within each final run directory listed at the start of this report, the principal supporting files are:

- `run_config_v3.json`
- `adapter_config.json`
- `adapter_model.safetensors` (header inspected only)
- `README.md`
- `checksums/SHA256SUMS.txt`
- the final checkpoint's `trainer_state.json`
- the final checkpoint's adapter, optimizer, scheduler, RNG, tokenizer, and training-argument artifacts

## Concise Thesis-Ready Summary

All six final models were obtained by 4-bit NF4 QLoRA fine-tuning of the same local `allenai/Olmo-3-1025-7B` base checkpoint. The frozen 7,298,011,136-parameter base model was augmented with 39,976,960 trainable BF16 LoRA parameters (`r=16`, `alpha=32`, dropout 0.05) applied to all attention projection and MLP projection modules. Training used plain-text flattened conversations, a 2,048-token maximum length, full-sequence causal-language-model loss, per-device batch size 1, 16 gradient-accumulation steps, and four distributed processes, giving an effective global batch size of 64. Each condition received approximately one pass over its own intervention dataset. Thus, optimization policy was held constant, but absolute training exposure varied with dataset size. The final checkpoint artifacts, trainer states, logs, and local checksum manifests support successful completion of all six runs, with no recorded non-finite losses.


---

# Appendix 7: IFEval completion and historical political readiness

Source: `EVALUATION_PROGRESS.md`.

# Evaluation progress and recovery handoff

IFEval recovery and official scoring are COMPLETE for all four adapters (2026-10-11). Final validation verified all 550 original responses unchanged, exactly 1,614 newly generated responses, and 541 unique benchmark prompt IDs per model (2,164 total). Adapter weights/configuration hashes and matched prompt/generation configurations passed verification.


| Model | Strict prompt | Loose prompt | Strict instruction | Loose instruction |
|---|---:|---:|---:|---:|
| Full | 56.56% | 60.63% | 66.79% | 70.02% |
| NoLeft | 57.49% | 62.11% | 66.67% | 70.50% |
| NoRight | 58.04% | 62.85% | 67.15% | 71.58% |
| LRBalanced | 54.71% | 58.41% | 64.39% | 67.99% |

All models use the same pinned official benchmark, prompt format, generation settings and seed. Exact counts, versions and file hashes are in comparison.json. These scores do not by themselves establish causal significance or broad capability.

Interrupted original segments lack trustworthy final runtime and CUDA peak measurements. Full-run totals are unavailable; resumed segment measurements are preserved separately. Political pilot scores remain diagnostic and are not final thesis results.

Official scorer caveat: keys 1122 and 1129 request punctuation frequencies ("#" and "!"), but the pinned letter-frequency checker accepts only ASCII letters and substitutes seeded random letters. Strict and loose passes can choose different fallback letters, so an official strict pass can become a loose failure. These are 2 of 834 instructions. Scores here preserve the official outputs without correction. Per-model anomalies are retained in summary.json and per_example.jsonl.

## Artifacts and recovery commands

- Thesis comparison: `outputs/evaluation_recovery_20261011/final_comparison/report.md`, `comparison.csv`, `comparison.json`, and `provenance.json`.
- Final integrity audit: `outputs/evaluation_recovery_20261011/final_output_validation.json`; SHA256 manifest: `artifact_manifest.json` in that directory.
- Exact executed tmux launch and CPU recovery commands: `outputs/evaluation_recovery_20261011/executed_commands.md` and `executed_commands.json`. All GPU recovery workers finished naturally; no generation worker was restarted or duplicated.
- Original responses: `outputs/evaluation_stage_20261011/{Model}/ifeval/` remain unchanged.
- New suffixes and per-prompt checkpoints: `outputs/evaluation_recovery_20261011/{Model}/resumed/`.
- Complete merged raw generations, scorer inputs and metadata: `{Model}/complete/`; official outputs: `complete/scoring/`; all per-example results and scores: `{Model}/analysis/`. Separate logs and historical recovery events remain per model.

Official source commit: `e6890f85757dd84e27ca6df2dd30651dafad28e0`. Dataset SHA256: `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`. Configuration preserved from the smoke test: USER/ASSISTANT training wrapper, greedy generation, max_new_tokens=1280, seed 42, BF16 base with PEFT runtime FP32 adapter tensors. Exact effective configuration/package versions are retained in metadata. Official source, data and scorer were unchanged. Analysis-only strict/loose monotonicity assumptions were removed to preserve official punctuation-fallback outcomes; historical failed analyses and CPU recovery commands are preserved.

## ProbVAA and CPU preparation

All four candidate political pilots are complete: 288 generations/model, all 12 intended template variants included, matched prompts and distinct reproducible sample seeds. Parsing success: Full 91.67%, NoLeft 88.19%, NoRight 90.97%, LRBalanced 92.01%. Pilot scores are diagnostic, not thesis political results. Report: `outputs/evaluation_stage_20261011/pilot_report.md`.

CPU-only preparation is complete under `outputs/probvaa_preparation_20261011/`: 17,208 unique prompts (239 statements × 6 formulations × 12 variants), per-model pilot scoring audits, metadata and README. Twenty-seven CPU tests passed. Recovered templates/wrapper and unique variant/seed choices are reconstructed protocol choices; reference parsing and scoring formulas remain unchanged. The old reference runner does not implement the candidate configuration. Full-run repeated sample count remains undecided. Remaining methodological issues include first-label ambiguity, residual truncation/NA handling, reference-template reconstruction and sample-count selection.

No full ProbVAA evaluation was launched. No trained adapters were modified. No further GPU evaluation is required for the completed IFEval benchmark. If revisiting historical running states, inspect actual processes rather than trusting saved status JSON.

## 2026-10-12 final CPU ProbVAA preparation

Original27tests and5newtests pass. Corrected pilots revalidated: all12variants,64-token configuration,paired distinct seeds,reference NA denominator. Resumable candidate runner and unexecuted tmux commands prepared; noGPUjob launched. Full launch awaits repeat count, reconstructed-protocol acceptance and residual truncation/parser decision. Read outputs/probvaa_final_preparation_20261012/report.md and candidate_s2_commands.sh. Paired IFEval bootstrap results: outputs/probvaa_final_preparation_20261012/ifeval_paired/report.md. Existing scientific outputs and adapters unchanged.

## 2026-10-12 original-paper audit — NO-GO supersedes earlier readiness

Latest instruction requires paper/appendix comparability and approval before any additional GPU inference. Full evaluation remains unlaunched. Read `outputs/probvaa_paper_audit_20261012/REPORT.md`, `protocol_comparison.csv`, `coverage_summary.json`, and `SFT_F_baseline_validation_plan.json` before proceeding.

Verified original2024 paper and official repository/HF/OSF archive; also version-pinned later training-data paper v1/v3. All239 statements/1434 English formulations match release, but current SFT candidates select base/generation templates, not Table7 instruction templates. Thirty repetitions are specified by both papers; two is an adapted stage. Original2024 drops invalid responses before reliability bootstrap; later notebook/current primary retains NA in denominator; later-v1 paper states parsed-only and magnitude-weighted policies. Do not conflate these scoring targets. Original release confirms a duplicated reversed instruction template (variants2/14), giving11 distinct prompt strings despite12 nominal slots. This needs an explicit paper-intent versus released-error decision.

Machine-readable full statement/prompt/appendix coverage and separate generated/parsed/NA/ambiguous/missing accounting saved. Paper-target SFT-F responses0/516240; candidate corrected pilot288responses with264parsed/24NA. Original parser matches1161test responses. All35 CPUtests and complete prompt tokenization passed.

Earlier authorized smoke is CHECKPOINTED at3/8responses in `outputs/probvaa_method_validation_20261012/Full_smoke`; it exited naturally and has no active worker. Do not resume without new GPU approval. Resume/end-to-end smoke remains untested. Supplementary reporting implemented in `code/evaluation/probvaa_supplementary.py`; outputs stay separate from primary. GPU4–7idle and396GiBfree at audit. Trained adapters and prior evaluation outputs unchanged.

Next: freeze original2024 vs later-paper primary target; declare repetition/prompt/order/decoding/format/scoring corrections; CPU-validate named protocol; then request approval for the prepared representative SFT-F GPU test. Do NOT execute prior candidate full-launch commands. Approx30-repeat runtime with unchanged64tokenbehavior is434–517GPUh/model;20token original-code scenario≈180–186h/model is uncertain extrapolation requiring measured smoke throughput.


---

# Appendix 8: Historical benchmark and classifier results

Source: `BENCHMARK_RESULTS_README.md`.

# Benchmark and Classifier Results Summary

This file consolidates the benchmark, classifier, and intervention results found so far. It is meant to answer:

- Which dataset was used?
- Which model or method was tested?
- What result did we get?
- What did that result imply?
- What was the next step?

## 1. Overall Progression

The project moved through this sequence:

```text
Manifesto benchmark construction
  -> prompted LLM benchmark runs
  -> encoder benchmark runs
  -> external clean ideology benchmarks
  -> SFT political candidate filtering
  -> SFT manual review and 4-way classification
  -> ModernBERT political/non-political gate
  -> OOD evaluation
  -> DOLCI/OLMo-3 SFT classification
  -> DOLCI intervention variants
```

The key lesson is that political classification is easy on clean text-level labels, but much harder on real SFT data and manifesto chunks where labels are noisy or context-dependent.

## 2. Manifesto Benchmark Results

### Dataset

Manifesto Project / MARPOR data from:

```text
data/raw/manifesto/MPDataset_MPDS2025a.csv
data/raw/manifesto/corpus/large_300_manifesto_corpus.csv
```

Labels were created by using the Manifesto-provided `rile` score:

```text
rile < -5   -> left
-5 to +5    -> center
rile > +5   -> right
```

The labels are document-level labels that were propagated to chunks.

### Benchmark v1 and v2 Prompted LLM Results

| Benchmark | Dataset | Rows | Documents | Model | Type | Chunk Accuracy | Document-Majority Accuracy | Main Pattern |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| benchmark_v1 | Manifesto chunks | 126 | 27 | Llama-3.3-70B-bnb-4bit | Prompted LLM | 0.6905 | 0.7407 | Best document-level result |
| benchmark_v1 | Manifesto chunks | 126 | 27 | Gemma-4-31B-AWQ | Prompted LLM | 0.6905 | 0.6667 | Similar chunk accuracy, stronger on right |
| benchmark_v1 | Manifesto chunks | 126 | 27 | Qwen3.5-27B-AWQ | Prompted LLM | 0.3333 | 0.3333 | Collapsed to all-left |
| benchmark_v2 | Manifesto chunks | 900 | 193 | Llama-3.3-70B-bnb-4bit | Prompted LLM | 0.5156 | 0.5440 | Overpredicted center |
| benchmark_v2 | Manifesto chunks | 900 | 193 | Gemma-4-31B-AWQ | Prompted LLM | 0.5044 | 0.5337 | Overpredicted right |
| benchmark_v2 | Manifesto chunks | 900 | 193 | Qwen3.5-27B-AWQ | Prompted LLM | 0.3333 | 0.3523 | Collapsed to all-left |

### Reasoning

Benchmark v2 is larger and more diverse than v1, so the drop from v1 to v2 is expected. The main issue is that RILE labels are document-level, but the model sees chunks. A chunk may not fully represent the party manifesto's overall ideology.

### Next Step

Use Manifesto benchmarks as an initial sanity check, but do not rely on them as the only evaluation. Move to cleaner text-level external benchmarks and target-domain SFT manual labels.

## 3. Manifesto Encoder Results

### Dataset

```text
data/processed/benchmark_v2/benchmark_v2.csv
```

Task:

```text
left / center / right
```

### Results

| Dataset | Model | Type | Test Setting | Chunk Accuracy | Macro F1 | Document-Majority Accuracy | Pattern |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| benchmark_v2 | Llama-3.3-70B-bnb-4bit | Prompted LLM | Full set | 0.5156 | n/a | 0.5440 | Overpredicted center |
| benchmark_v2 | Gemma-4-31B-AWQ | Prompted LLM | Full set | 0.5044 | n/a | 0.5337 | Overpredicted right |
| benchmark_v2 | Qwen3.5-27B-AWQ | Prompted LLM | Full set | 0.3333 | n/a | 0.3523 | All-left collapse |
| benchmark_v2 | ModernBERT-large | Fine-tuned encoder | Held-out document split | 0.4194 | 0.4190 | 0.4000 | Weak, not collapsed |
| benchmark_v2 | DeBERTa-v3-large | Fine-tuned encoder | Held-out document split | 0.3629 | 0.1775 | 0.3333 | All-right collapse |
| benchmark_v2 | mDeBERTa-v3-base | Fine-tuned encoder | Held-out document split | 0.3065 | 0.1564 | 0.3333 | All-center collapse |

### Reasoning

Encoders did not solve the Manifesto chunk-label problem. ModernBERT was weak but did not collapse; DeBERTa and mDeBERTa collapsed to a single class. This reinforced that weak document labels copied to chunks are a major limitation.

### Next Step

Create or use benchmarks with cleaner text-level labels.

## 4. External Clean Ideology Benchmarks

## 4.1 Benchmark v3a: Clean Left/Right Ideology

### Dataset

```text
JyotiNayak/political_ideologies
data/processed/benchmark_v3/benchmark_v3a_political_ideologies_binary.*
```

Task:

```text
left / right
```

### Results

| Dataset | Model | Type | Test Rows | Parsed Rows | Parse Rate | Accuracy All Rows | Accuracy Parsed Only | Macro F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| benchmark_v3a | ModernBERT-large | Fine-tuned encoder | 320 | 320 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Perfect on clean benchmark |
| benchmark_v3a | Llama-3.3-70B-bnb-4bit | Prompted LLM | 320 | 320 | 1.0000 | 0.9437 | 0.9437 | n/a | Strong, all outputs parsed |
| benchmark_v3a | Qwen3.5-27B-AWQ | Prompted LLM | 320 | 238 | 0.7438 | 0.7312 | 0.9832 | n/a | Strong on parsed rows, parsing failures from reasoning tags |
| benchmark_v3a | DeBERTa-v3-large | Fine-tuned encoder | 320 | 320 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | Collapsed to all-left |
| benchmark_v3a | mDeBERTa-v3-base | Fine-tuned encoder | 320 | 320 | 1.0000 | 0.5000 | 0.5000 | 0.3333 | Collapsed to all-right |
| benchmark_v3a | Gemma-4-31B-AWQ | Prompted LLM | 320 | 67 | 0.2094 | 0.1437 | 0.6866 | n/a | Unstable output format |

### Reasoning

When labels are clean and text-level, ModernBERT and Llama perform very well. This means the models can learn political distinctions when the benchmark is well aligned with the task.

### Next Step

Use clean benchmarks to validate model capacity, but move back to SFT data where the real challenge is separating political content from non-political examples with political words.

## 4.2 Benchmark v3b: Senator Tweets

### Dataset

```text
m-newhauser/senator-tweets
```

Task:

```text
Democrat(left) / Republican(right)
```

### Results

| Dataset | Model | Type | Test Rows | Accuracy | Macro F1 | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| benchmark_v3b senator tweets | ModernBERT-large | Fine-tuned encoder | 320 | 0.8688 | 0.8687 | Stronger than prompted Llama on harder real-world political text |
| benchmark_v3b senator tweets | Llama-3.3-70B-bnb-4bit | Prompted LLM | 320 | 0.8344 | n/a | Parsed all rows, overpredicted left |

### Reasoning

This was a more realistic text-level political benchmark than v3a. ModernBERT performed well, suggesting an encoder can be useful as a fast classifier when trained on appropriate labels.

### Next Step

Train a target-domain encoder on manually labeled SFT rows.

## 5. SFT 4-Way Manual Review Results

### Dataset

Tulu/OLMo-2 SFT candidate sample:

```text
outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_labeled.csv
```

Task:

```text
0 = non_political
1 = left
2 = center_neutral
3 = right
```

### 4-Way LLM Classifier Result

| Dataset | Model | Type | Evaluated Rows | Accuracy | Main Pattern |
| --- | --- | --- | ---: | ---: | --- |
| Tulu/OLMo-2 SFT manual sample | Llama-3.3-70B-bnb-4bit | Prompted 4-way LLM | 192 | 0.5833 | High sensitivity, overpredicts political labels on non-political examples |

Per-class behavior:

| Gold Class | Recall | Notes |
| --- | ---: | --- |
| non_political | 0.4757 | Many non-political examples with political words were called political |
| left | 0.9062 | Strong recall |
| center_neutral | 0.5833 | Moderate recall |
| right | 0.8571 | Strong recall but small support |

### Binary Political/Non-Political View

| Dataset | Model | Rows | Accuracy | Non-Political Recall | Political Recall | Main Pattern |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Tulu/OLMo-2 SFT manual sample | Llama-3.3-70B-bnb-4bit 4-way collapsed to binary | 187 | 0.7219 | 0.4900 | 0.9885 | Very high political recall but many false positives |

### Reasoning

The LLM classifier is useful for high recall. It almost never misses political examples in the binary view, but it over-flags non-political examples such as math, translation, and safety refusals that contain political words.

### Next Step

Train a cheaper target-domain binary gate, ModernBERT, using manual labels.

## 6. Target-Domain ModernBERT Gate Results

### Dataset

Manual SFT labels converted to binary:

```text
non_political
political
```

### Results

| Dataset | Model | Type | Rows | Accuracy | Non-Political Recall | Political Recall | Confusion Matrix | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Initial target-domain split | ModernBERT-base | Fine-tuned encoder | 52 | 0.9423 | 0.9000 | 1.0000 | `[[27, 3], [0, 22]]` | Very strong, but small test split |
| Combined target-domain split | ModernBERT-base / combined | Fine-tuned encoder | 127 | 0.8898 | 0.9125 | 0.8511 | `[[73, 7], [7, 40]]` | More realistic, still useful |

### Reasoning

ModernBERT is much cheaper than an LLM and performs well on in-domain manual splits. This supports using it as stage 1 in a two-stage classifier.

### Next Step

Evaluate whether the gate generalizes out of distribution. This is necessary because stage 1 false negatives cannot be recovered by stage 2.

## 7. ModernBERT OOD Evaluation

### Dataset

```text
outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv
```

Rows with gold labels:

```text
299
```

Gold distribution:

| Label | Rows |
| --- | ---: |
| non_political | 196 |
| center_neutral | 98 |
| right | 5 |

### Result

| Dataset | Model | Type | Rows | Accuracy | Macro F1 | Non-Political Recall | Political Recall | Confusion Matrix |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OOD SFT manual sample | ModernBERT target-domain gate | Binary encoder | 299 | 0.6756 | 0.6181 | 0.8112 | 0.4175 | `[[159, 37], [60, 43]]` |

### Reasoning

The OOD test revealed the main weakness: ModernBERT misses many neutral political examples. Since the two-stage pipeline only runs the expensive LLM classifier on rows that pass the gate, false negatives at stage 1 are unrecoverable.

### Next Step

Improve or recalibrate the gate, especially for neutral political examples. The DOLCI branch uses the combined gate but still treats results as model-generated labels requiring caution.

## 8. DOLCI/OLMo-3 SFT Classification Results

### Dataset

```text
allenai/Dolci-Instruct-SFT-7B
allenai/Dolci-Think-SFT-7B
```

Flattened files:

```text
data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl
data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl
```

Pipeline:

```text
DOLCI rows
  -> combined ModernBERT binary gate
  -> Llama-3.3-70B 4-way classifier on binary candidates
  -> intervention row lists
```

### 4-Way Candidate Classification Summary

| Label | Rows |
| --- | ---: |
| non_political | 88,634 |
| center_neutral | 60,916 |
| left | 28,861 |
| right | 3,955 |
| unparsed | 5,251 |
| total binary candidates | 187,617 |

By split:

| Label | Instruct Rows | Think Rows |
| --- | ---: | ---: |
| non_political | 70,375 | 18,259 |
| center_neutral | 38,146 | 22,770 |
| left | 16,859 | 12,002 |
| right | 2,833 | 1,122 |
| unparsed | 3,533 | 1,718 |

Key derived counts:

| Quantity | Value |
| --- | ---: |
| LLM-confirmed political: `center_neutral + left + right` | 93,732 |
| Ideological: `left + right` | 32,816 |
| Left rows | 28,861 |
| Right rows | 3,955 |
| Left/right ratio | 7.30 : 1 |

### Reasoning

DOLCI ideological examples are strongly left-heavy under the current classifier. This motivated creating multiple intervention variants instead of only one removal dataset.

### Next Step

Build intervention datasets for controlled fine-tuning experiments.

## 9. DOLCI/OLMo-3 Intervention Variants

### Dataset

Original total:

| Split | Rows |
| --- | ---: |
| instruct | 2,152,112 |
| think | 2,268,178 |
| total | 4,420,290 |

### Variants Built

| Variant | Removes | Instruct Removed | Think Removed | Total Removed | Total Kept | Purpose |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `no_pol` | `center_neutral`, `left`, `right` | 57,838 | 35,894 | 93,732 | 4,326,558 | Remove all LLM-confirmed political content |
| `no_ideology` | `left`, `right` | 19,692 | 13,124 | 32,816 | 4,387,474 | Remove ideological content but keep neutral political explanations |
| `no_left` | `left` | 16,859 | 12,002 | 28,861 | 4,391,429 | Test effect of removing left-labeled examples |
| `no_right` | `right` | 2,833 | 1,122 | 3,955 | 4,416,335 | Test effect of removing right-labeled examples |
| `lr_balanced` | extra `left` rows | 14,550 | 10,356 | 24,906 | 4,395,384 | Downsample left so remaining left/right counts are balanced |

### Reasoning

Different interventions answer different causal questions:

- `no_pol`: What if all detected political content is removed?
- `no_ideology`: What if only ideological positioning is removed?
- `no_left`: What if left-labeled examples are removed?
- `no_right`: What if right-labeled examples are removed?
- `lr_balanced`: What if the ideological distribution is balanced instead of removed?

### Next Step

Fine-tune models on each intervention variant and evaluate whether downstream behavior changes.

## 10. Fine-Tuning Variants So Far

### Dataset and Model

Base model:

```text
hf_cache/allenai_OLMo-2-1124-7B
```

Training style:

```text
QLoRA / LoRA SFT using TRL SFTTrainer
```

### Runs

| Run | Data | Script | Output | Purpose |
| --- | --- | --- | --- | --- |
| smoke 5k baseline | `baseline_original_smoke_5k_textonly.jsonl` | `code/train_olmo2_lora_sft_smoke.py` | `outputs/sft_runs/variant_01_full_original_smoke_5k_lora/` | Confirm training setup works |
| 100k baseline | `variant_01_full_original_sft_100k_textonly.jsonl` | `code/train_olmo2_lora_sft_variant01_100k.py` | `outputs/sft_runs/variant_01_full_original_sft_100k_lora/` | Larger baseline run before intervention runs |

### Reasoning

These are baseline training runs. They prove the LoRA SFT stack can run before spending compute on all intervention variants.

### Next Step

Run equivalent fine-tuning jobs for DOLCI intervention datasets, then compare model behavior across baseline and variants.

## 11. Current Interpretation

The project has found three important things:

1. Clean ideology classification is feasible.

ModernBERT and Llama perform strongly on clean text-level left/right benchmarks.

2. Real SFT political filtering is harder.

SFT data contains many examples with political words that are not political-position examples. Manual review and OOD evaluation are necessary.

3. DOLCI political candidates are left-heavy.

Among DOLCI ideological examples classified so far, left-labeled rows outnumber right-labeled rows by about 7.30 to 1.

## 12. Recommended Next Steps

1. Treat Manifesto chunk results as diagnostic, not final proof.
2. Improve or recalibrate the ModernBERT gate using OOD findings.
3. Keep using Llama-3.3-70B for 4-way classification when richer labels are needed.
4. Fine-tune models on the DOLCI intervention variants.
5. Evaluate downstream behavior across:
   - baseline
   - `no_pol`
   - `no_ideology`
   - `no_left`
   - `no_right`
   - `lr_balanced`
6. Report not only classifier metrics, but also how model behavior changes after data intervention.

## 13. Source Files Used

| Result Area | Source Artifact |
| --- | --- |
| Manifesto v1/v2 prompted results | `outputs/original_classifier/final_comparison/v1_v2_benchmark_comparison.md` |
| Benchmark v2/v3 comparison | `outputs/classifier_comparison/classifier_benchmark_summary_v2_v3.md` |
| All classifier comparison | `outputs/classifier_comparison/all_classifier_comparison.md` |
| Benchmark v3a final summary | `outputs/benchmark_v3a_binary/summary/benchmark_v3a_final_summary.md` |
| SFT manual review labels | `outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_labeled.csv` |
| Target-domain ModernBERT test predictions | `outputs/target_domain_binary_modernbert/test_predictions.csv` |
| Combined ModernBERT test predictions | `outputs/target_domain_binary_modernbert_combined/test_predictions.csv` |
| OOD ModernBERT summary | `outputs/modernbert_ood_test/two_stage_eval/two_stage_ood_summary.json` |
| DOLCI 4-way summary | `outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_4way_summary.json` |
| DOLCI intervention manifest | `data/interventions/dolci_olmo3_sft/intervention_manifest.json` |



---

# Appendix 9: Historical pipeline chronology and workflows

Source: `CHRONOLOGICAL_PROJECT_README.md`.

# Chronological Project README

This project is a research pipeline for understanding and controlling political
content in instruction-tuning data. The core question is:

> Can we identify political examples inside a large SFT mixture, understand
> their ideological balance, and build interventions that reduce or rebalance
> political content without blindly removing useful data?

The project grows in stages. First, it builds external political benchmarks so
we can measure classifiers. Then it inspects and filters the Tulu/OLMo SFT
mixture. Then it manually labels samples, trains a target-domain ModernBERT
political-content gate, runs a second-stage 4-way ideology classifier, evaluates
OOD behavior, and finally creates SFT intervention datasets and LoRA training
runs. The newer active branch applies the same detection/intervention idea to
OLMo-3 DOLCI SFT data, treating `Dolci-Instruct-SFT-7B` and
`Dolci-Think-SFT-7B` as separate SFT splits throughout flattening,
classification, token accounting, and intervention JSONL creation.

## Environment

Run most Python scripts from the project root:

```bash
cd /fs/scratch/users/bharatpi/causal_political_content
```

The main virtual environment is:

```bash
source envs/classifier/bin/activate
```

Or call Python directly:

```bash
envs/classifier/bin/python code/<script>.py
```

Many scripts assume local caches and outputs under:

- `data/raw/`
- `data/processed/`
- `outputs/`
- `cache/huggingface/`
- `hf_cache/`

## Label Conventions

The project uses several related label spaces:

| Task | Labels |
| --- | --- |
| Manifesto 3-way ideology | `left`, `center`, `right` |
| Numeric 3-way ideology | `1 = left`, `2 = center/neutral`, `3 = right` |
| SFT 4-way political label | `0 = non_political`, `1 = left`, `2 = center_neutral`, `3 = right` |
| ModernBERT political gate | `0 = non_political`, `1 = political` |
| Binary ideology benchmarks | usually `left` / `right` |

## Chronological Story

### 1. Establish Political Classification Benchmarks

The first problem is measurement. Before changing SFT data, we need benchmark
tasks where political orientation can be evaluated. This part uses Manifesto
Project data and public political-bias datasets.

The Manifesto work creates left/center/right examples from party manifestos.
The external v3 benchmarks add binary ideology tasks from non-manifesto sources,
which helps test whether classifiers generalize beyond one domain.

### 2. Run Prompted LLM Classifiers

After building benchmarks, the project runs LLMs as political classifiers. These
experiments test prompt design, parsing reliability, model agreement, and whether
large instruction models can classify political text consistently.

This stage motivates the move from a simple 3-way political prompt to a 4-way
prompt that explicitly includes `non_political`. That distinction matters because
SFT mixtures contain many coding, math, table, translation, and safety examples
that may mention political words without being politically classifiable.

### 3. Train Encoder Classifiers

Prompted LLMs are expensive. The next stage trains encoder classifiers for
faster, repeatable scoring. There are two flavors:

- Manifesto/domain benchmark encoders for left/center/right or left/right tasks.
- A target-domain ModernBERT binary gate trained on manually labeled SFT samples.

The ModernBERT gate becomes stage 1 of the two-stage classifier:

```text
SFT example
  -> Stage 1: ModernBERT political/non-political gate
  -> Stage 2: LLM 4-way ideology classifier on examples that pass the gate
```

### 4. Inspect and Flatten the Tulu/OLMo SFT Mixture

The project then turns to the actual target domain: the Tulu/OLMo SFT mixture.
The mixture is inspected, flattened into row-level conversations, and filtered
with political keyword heuristics to create candidate pools.

This is intentionally broad at first. The goal is recall: find possible
political examples, then narrow them with model classifiers and manual labels.

### 5. Manual Labeling and Training Data Creation

Manual review is used to establish ground truth for SFT examples. The labels
distinguish truly non-political tasks from neutral political explanations and
left/right ideological content.

The initial labeled sample is used both to evaluate the 4-way LLM classifier and
to train the target-domain ModernBERT political gate.

### 6. Apply ModernBERT and Evaluate OOD Behavior

After training the binary gate, the project applies ModernBERT across candidate
SFT rows and separately tests it on a new 300-row OOD manual sample.

This stage revealed an important behavior: the gate can over-filter neutral
political examples. That matters because a two-stage classifier cannot recover
political examples that stage 1 rejects.

### 7. Analyze Errors and Inspect OOD Failures

The OOD analysis tools split errors into false positives and false negatives,
then support manual review of the most consequential mistakes. This helps answer
whether the model is wrong, the label guideline is ambiguous, or the task needs
a different threshold/architecture.

### 8. Build Tulu/OLMo-2 SFT Interventions

Once political-position examples are detected, the project creates intervention
datasets:

- remove all detected political-position rows
- rebalance left/right examples
- downsample overrepresented groups
- upsample underrepresented groups

These produce full SFT JSONL variants that can be used for controlled LoRA
fine-tuning experiments.

### 9. Train OLMo-2 LoRA Variants

This stage trains OLMo-2 LoRA variants on baseline or intervention data.
The long-term objective is to compare how different SFT data interventions
change downstream behavior.

### 10. Extend the Pipeline to OLMo-3 DOLCI SFT

The newer branch moves from the earlier Tulu/OLMo-2 mixture to OLMo-3 DOLCI SFT
datasets:

- `allenai/Dolci-Instruct-SFT-7B`
- `allenai/Dolci-Think-SFT-7B`

These are handled as two splits, `instruct` and `think`. They are flattened into
row-level JSONL, passed through the combined ModernBERT gate, prepared for the
LLM 4-way classifier, and then joined into a combined DOLCI classification file.

The combined 4-way labels are used to build row-removal lists:

- `no_pol`: remove `center_neutral`, `left`, and `right`
- `no_ideology`: remove only `left` and `right`
- `no_left`: remove left-labeled rows
- `no_right`: remove right-labeled rows
- `lr_balanced`: remove extra left rows so left/right counts are balanced

The DOLCI branch also adds token accounting. It counts tokens by label and split
using the Llama-3.3 tokenizer, and separately estimates how many tokens each
intervention variant removes from the full `instruct` + `think` corpus.

### 11. Build OLMo-3 DOLCI Intervention JSONLs

After row lists are created, the project streams through both DOLCI flattened
SFT files and writes intervention JSONL variants under:

```text
data/interventions/dolci_olmo3_sft/jsonl/
```

Each variant keeps separate output files for the `instruct` and `think` splits,
so later training/evaluation can preserve or explicitly control the original
DOLCI split identity.

## Current Two-Stage Classifier Path

The earlier Tulu/OLMo-2 OOD path is:

```text
outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv
  -> code/evaluate_modernbert_on_ood_sample.py
  -> outputs/modernbert_ood_test/modernbert_ood_predictions.csv
  -> code/evaluate_two_stage_ood.py
  -> outputs/modernbert_ood_test/two_stage_eval/
```

To complete full two-stage OOD scoring, run the 4-way classifier on the same OOD
rows and pass that output into `evaluate_two_stage_ood.py` with `--stage2`.

The newer OLMo-3 DOLCI path is:

```text
allenai/Dolci-Instruct-SFT-7B
allenai/Dolci-Think-SFT-7B
  -> code/download_inspect_olmo3_dolci_sft.py
  -> code/flatten_olmo3_dolci_sft.py
  -> code/apply_modernbert_to_dolci_olmo3_sft.py
  -> code/prepare_dolci_candidates_for_4way.py
  -> code/run_sft_political_4way_classifier.py
  -> outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/
  -> code/create_dolci_olmo3_intervention_row_lists.py
  -> code/compute_dolci_olmo3_token_stats_by_label.py
  -> code/compute_dolci_intervention_token_removal_stats.py
  -> code/build_dolci_olmo3_intervention_jsonl.py
  -> data/interventions/dolci_olmo3_sft/jsonl/
```

## Python Files in Chronological Order

The sections below list every Python file in `apps/` and `code/`, ordered by the
pipeline stage where it naturally belongs.

## Stage 0: Shared Utilities and Prompt Definitions

These files are used by multiple later stages.

| File | Role |
| --- | --- |
| `code/benchmark_utils.py` | Shared helper functions for project path resolution, JSON writing, timestamping, model-name sanitization, label normalization, and parsing numeric/label outputs from models. |
| `code/benchmark_prompts.py` | Stores prompt templates for Manifesto benchmark classification and exposes `build_prompt()` for runners. |
| `code/test_model_load.py` | Small smoke test for checking whether a local or Hugging Face causal LM can load and generate. Useful before launching expensive runs. |

## Stage 1: Manifesto Data Preparation

These scripts build the first controlled political benchmark from Manifesto
Project data.

| File | Role |
| --- | --- |
| `code/select_manifestos_for_large_download.py` | Reads Manifesto metadata, maps RILE scores into left/center/right labels, and samples a balanced list of manifestos for later corpus download. |
| `code/chunk_manifestos.py` | Splits a smaller downloaded Manifesto text sample into sentence-aware chunks. It is an early chunking script. |
| `code/chunk_manifestos_strict.py` | Creates stricter word-count-based chunks from Manifesto text, giving more consistent benchmark rows. |
| `code/create_manifesto_large_chunks.py` | Builds overlapping chunks from the larger 300-document Manifesto corpus and writes large chunk-level data plus metadata. |
| `code/analyze_chunk_similarity.py` | Computes TF-IDF cosine similarity among chunks from the same document to detect repeated or overly similar chunks. |

## Stage 2: Manifesto Benchmark Construction

These scripts convert chunks into benchmark CSVs.

| File | Role |
| --- | --- |
| `code/build_benchmark_v1.py` | Creates benchmark v1 from prepared chunks, normalizes labels, balances classes if requested, adds prompt metadata, and can create splits. |
| `code/create_benchmark_v2_from_large_chunks.py` | Creates a larger balanced benchmark from the 300-document chunk pool, using similarity filtering to choose diverse chunks. |

## Stage 3: Manifesto Prompted Classifier Runs

These scripts run LLMs as left/center/right political classifiers on Manifesto
benchmarks.

| File | Role |
| --- | --- |
| `code/run_benchmark_v1.py` | General benchmark v1 runner supporting vLLM or Transformers backends, multiple models, resume mode, parsing, and error JSONs. |
| `code/run_llama_manifesto_classifier.py` | Earlier row-by-row Manifesto classifier. It includes dry-run and single-prompt test paths and is useful for debugging prompt/model behavior. |
| `code/run_original_classifier_on_benchmark.py` | Runs the original 3-way political-language prompt on a benchmark and saves parsed predictions. |

## Stage 4: Manifesto Evaluation and Error Analysis

These scripts summarize benchmark performance and inspect model disagreements.

| File | Role |
| --- | --- |
| `code/evaluate_benchmark_v1.py` | Evaluates benchmark v1 prediction CSVs and writes summary metrics, per-class metrics, confusion matrices, and a report. |
| `code/evaluate_benchmark_v2_document_majority.py` | Aggregates chunk-level benchmark v2 predictions into document-level majority predictions. |
| `code/analyze_benchmark_v2_errors.py` | Joins Llama, Gemma, and Qwen benchmark v2 outputs and writes detailed agreement/error subsets. |
| `code/analyze_benchmark_v2_errors_light.py` | Lighter benchmark v2 error-analysis variant that writes compact summaries and samples. |
| `code/create_final_benchmark_comparison.py` | Combines v1 and v2 original-classifier results into final comparison CSV/Markdown outputs. |
| `code/create_all_classifier_comparison.py` | Manually assembles a high-level comparison table across prompted LLM and encoder classifiers. |
| `code/create_classifier_benchmark_summary_v2_v3.py` | Summarizes classifier results across benchmark v2 and v3 into CSV and Markdown. |

## Stage 5: External Political Benchmark Discovery and Creation

These files search for and build non-Manifesto benchmarks, mainly for
left/right ideology classification.

| File | Role |
| --- | --- |
| `code/try_lcr_dataset_candidates.py` | Checks candidate Hugging Face datasets that might support left/center/right classification. |
| `code/inspect_political_datasets.py` | Loads and prints structure/examples from several political datasets to decide whether they are usable. |
| `code/inspect_news_political_bias.py` | Inspects a news political-bias dataset candidate. |
| `code/inspect_newsmediabias.py` | Inspects a media-bias dataset candidate. |
| `code/inspect_allsides_streaming.py` | Streams examples from an AllSides-style dataset for structure and label inspection. |
| `code/create_benchmark_v3a_political_ideologies.py` | Builds benchmark v3a from `JyotiNayak/political_ideologies`, mapping labels into left/right. |
| `code/create_benchmark_v3b_senator_tweets_binary.py` | Builds a binary benchmark from senator tweets, mapping Democrat/Republican party to left/right. |
| `code/create_benchmark_v3b_allsides.py` | Builds an AllSides-style news benchmark with political-bias labels. |

## Stage 6: External Binary Classifier Runs and Summaries

These scripts evaluate models on binary external political benchmarks.

| File | Role |
| --- | --- |
| `code/run_binary_political_classifier.py` | Runs a numeric left/right prompt over binary benchmarks and saves parsed predictions. |
| `code/run_binary_political_classifier_v2.py` | Runs a stricter left/right label prompt with improved parsing and reporting. |
| `code/reparse_binary_outputs.py` | Re-parses messy binary LLM outputs after the fact and writes `__reparsed.csv` files. |
| `code/create_benchmark_v3a_summary.py` | Summarizes prompted binary classifier outputs for benchmark v3a. |
| `code/create_benchmark_v3a_final_summary.py` | Combines prompted LLM and encoder results for benchmark v3a into final summary tables. |

## Stage 7: Encoder Training on Benchmark Data

These files train or audit encoder models for benchmark tasks.

| File | Role |
| --- | --- |
| `code/train_encoder_manifesto_classifier.py` | Trains a 3-way encoder classifier on Manifesto benchmark v2 with document-level splitting and saves chunk/document predictions. |
| `code/train_encoder_benchmark_v3a_binary.py` | Trains a binary left/right encoder on v3-style benchmark splits and writes test predictions plus summary metrics. |

## Stage 8: Inspect and Flatten the Tulu/OLMo SFT Mixture

These scripts move from external benchmarks to the target SFT data.

| File | Role |
| --- | --- |
| `code/inspect_tulu_olmo2_sft.py` | Streams quick examples from the Tulu/OLMo SFT mixture for initial inspection. |
| `code/inspect_tulu_olmo2_sft_full.py` | Loads the full mixture and writes source, dataset, role-pattern, and turn-count distributions. |
| `code/flatten_tulu_olmo2_sft.py` | Converts multi-message SFT examples into flat JSONL rows with combined text and sample files. |

## Stage 9: Create Candidate Political SFT Pools

These scripts find likely political examples in the flattened SFT mixture.

| File | Role |
| --- | --- |
| `code/prefilter_tulu_olmo2_political_candidates.py` | Uses broad political keyword filters to create a candidate pool from flattened SFT examples. |
| `code/create_high_confidence_political_candidates.py` | Applies stronger keyword/category rules to create a smaller high-confidence candidate pool. |
| `code/create_tulu_sft_highconf_classification_sample.py` | Samples high-confidence candidates into CSV/JSONL files for classifier runs and manual review. |

## Stage 10: 4-Way SFT Political Classification

This stage asks an LLM to classify SFT conversations into non-political, left,
center/neutral, or right.

| File | Role |
| --- | --- |
| `code/run_sft_political_4way_classifier.py` | Runs the 4-way SFT political prompt using vLLM and saves raw outputs plus parsed labels. |
| `code/reparse_sft_4way_outputs.py` | Re-parses 4-way classifier raw outputs with numeric and textual fallbacks. |
| `code/compare_original_vs_our_prompt.py` | Compares the older 3-way prompt against the newer 4-way prompt on a labeled file. |
| `code/compare_original_vs_our_prompt_gpus23.py` | GPU-specific variant of the prompt comparison script. |
| `code/compare_original_prompt_three_models.py` | Runs the original 3-way prompt across three configured models for comparison. |

## Stage 11: Manual Review Tooling

Manual labels are central because heuristic filters and LLM classifiers are not
trusted as ground truth.

| File | Role |
| --- | --- |
| `code/create_sft_4way_manual_review_sample.py` | Samples classifier outputs into a manual-review CSV, usually stratified to inspect model predictions. |
| `code/export_manual_review_to_txt.py` | Exports manual-review rows into a readable text batch for external labeling or review. |
| `code/import_manual_labels_from_tsv.py` | Merges externally prepared TSV labels back into the review CSV. |
| `apps/manual_sft_labeler.py` | Streamlit app for labeling SFT examples as `0/1/2/3/U`, saving progress into the labeled CSV. |
| `code/evaluate_manual_review_sample.py` | Evaluates 4-way classifier predictions against manual labels and writes error cases. |
| `code/create_binary_manual_eval.py` | Converts 4-way manual labels/predictions into a binary political/non-political or left/right evaluation view. |
| `code/evaluate_binary_manual_review.py` | Evaluates the binary manual-review file. |

## Stage 12: Train the Target-Domain ModernBERT Gate

This stage creates the first part of the two-stage classifier: a fast binary
political/non-political gate trained on manually labeled SFT examples.

| File | Role |
| --- | --- |
| `code/create_target_domain_binary_train.py` | Converts the manually labeled 4-way review file into `target_domain_binary_manual_train.csv`, mapping `0` to non-political and `1/2/3` to political. |
| `code/check_modernbert_train_val_split.py` | Audits the target-domain training file and expected train/validation split assumptions before training. |
| `code/train_encoder_target_domain_binary.py` | Trains `answerdotai/ModernBERT-base` as a binary political-content classifier and saves `outputs/target_domain_binary_modernbert/best_model`. |
| `code/apply_target_domain_binary_modernbert.py` | Applies the trained ModernBERT gate to SFT political candidates and writes probabilities/predictions. |

## Stage 13: Two-Stage Prompting After ModernBERT Filtering

This stage evaluates the full idea of using a cheap gate before an expensive
LLM ideology classifier.

| File | Role |
| --- | --- |
| `code/compare_prompts_after_modernbert_filter.py` | Merges manual labels with ModernBERT predictions, runs prompts only on rows that pass the gate, and evaluates final two-stage outcomes. |

## Stage 14: OOD Manual Sample and ModernBERT OOD Evaluation

This stage tests whether the ModernBERT gate trained on one labeled sample
generalizes to a new manually labeled OOD sample.

| File | Role |
| --- | --- |
| `code/create_modernbert_ood_manual_sample.py` | Samples 300 new SFT candidate rows excluding the earlier manual-review row IDs, creating an OOD manual-labeling file. |
| `code/evaluate_modernbert_on_ood_sample.py` | Runs the trained ModernBERT gate on the 300-row labeled OOD sample and reports binary metrics. |
| `code/analyze_modernbert_ood_errors.py` | Splits ModernBERT OOD predictions into correct, false-positive, and false-negative CSVs plus focused high-confidence error samples. |
| `apps/modernbert_ood_error_inspector.py` | Streamlit app for inspecting ModernBERT OOD errors and recording review decisions/notes. |
| `code/evaluate_two_stage_ood.py` | Evaluates the OOD sample in stage-1-only mode, or full two-stage mode when a 4-way stage-2 prediction file is supplied. |

## Stage 15: Build Tulu/OLMo-2 SFT Intervention Datasets

After detecting political-position examples, this stage builds modified SFT
datasets for causal/intervention experiments.

| File | Role |
| --- | --- |
| `code/create_sft_intervention_jsonl.py` | Creates full SFT JSONL variants: baseline, remove detected political-position rows, downsample left, and upsample right. |

## Stage 16: Train OLMo-2 LoRA Variants

These scripts fine-tune OLMo-2 on baseline/intervention data.

| File | Role |
| --- | --- |
| `code/train_olmo2_lora_sft_smoke.py` | Runs a small LoRA SFT smoke test on a 5k text-only JSONL file. |
| `code/train_olmo2_lora_sft_variant01_100k.py` | Runs the larger 100k-example LoRA SFT training variant. |

## Stage 17: Inspect and Flatten OLMo-3 DOLCI SFT

This stage extends the project from Tulu/OLMo-2 to the newer OLMo-3 DOLCI SFT
datasets. There are two source datasets, handled as separate splits:

| DOLCI dataset | Local split name | Meaning |
| --- | --- | --- |
| `allenai/Dolci-Instruct-SFT-7B` | `instruct` | Standard instruction-following SFT data. |
| `allenai/Dolci-Think-SFT-7B` | `think` | Reasoning/think-style SFT data, generally much longer per row. |

| File | Role |
| --- | --- |
| `code/download_inspect_olmo3_dolci_sft.py` | Loads both DOLCI datasets, prints row/column structure, and writes small preview JSONL files under `data/raw/olmo3_dolci_sft/`. |
| `code/flatten_olmo3_dolci_sft.py` | Converts DOLCI message lists into flat row-level JSONL files under `data/processed/dolci_olmo3_sft/`, preserving `row_id`, dataset name, source metadata, messages, and combined text. |

The main flattened outputs are:

```text
data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl
data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl
```

## Stage 18: Apply the Combined ModernBERT Gate to DOLCI

This stage applies the trained/combined target-domain ModernBERT binary gate to
the DOLCI `instruct` and `think` flattened files. It writes a metadata-only
prediction JSONL and a full-row political candidate JSONL.

| File | Role |
| --- | --- |
| `code/train_encoder_target_domain_binary_combined.py` | Trains the later combined ModernBERT binary gate and saves `outputs/target_domain_binary_modernbert_combined/best_model`. |
| `code/evaluate_combined_modernbert_on_corrected_ood.py` | Evaluates the combined ModernBERT gate on corrected OOD labels. |
| `code/apply_modernbert_to_dolci_olmo3_sft.py` | Applies the combined ModernBERT model to a flattened DOLCI JSONL, using a configurable probability threshold, and writes candidate rows where `prob_political >= threshold`. |

The important point is that this is a model gate, not a keyword-only filter. It
is used to avoid sending every DOLCI row to the much more expensive 4-way LLM
classifier.

## Stage 19: Prepare and Classify DOLCI Candidates with the 4-Way LLM Prompt

Rows that pass the DOLCI ModernBERT gate are converted into CSV inputs for the
same 4-way SFT political classifier used earlier.

| File | Role |
| --- | --- |
| `code/prepare_dolci_candidates_for_4way.py` | Converts DOLCI candidate JSONL files for both `instruct` and `think` into CSV files with `full_text`, ModernBERT probability, IDs, and metadata expected by the 4-way classifier. |
| `code/run_sft_political_4way_classifier.py` | Runs the 4-way prompt on DOLCI candidate CSVs, producing raw outputs and parsed labels. |
| `code/reparse_sft_4way_outputs.py` | Can be reused if DOLCI 4-way outputs need numeric/textual reparsing. |

The combined DOLCI classification file used by later stages is:

```text
outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_binary_candidates_4way_combined.csv
```

Its central label column is:

```text
final_political_4way_status
```

with values like `non_political`, `center_neutral`, `left`, `right`, or missing
values treated as `unparsed`.

## Stage 20: Create DOLCI Intervention Row Lists

This stage turns the combined DOLCI 4-way classification file into row-removal
lists for intervention experiments.

| File | Role |
| --- | --- |
| `code/create_dolci_olmo3_intervention_row_lists.py` | Reads the combined DOLCI 4-way classification file, normalizes labels, and writes row-ID lists for `no_pol`, `no_ideology`, `no_left`, `no_right`, and `lr_balanced` interventions. |

The intervention meanings are:

| Variant | Removes |
| --- | --- |
| `no_pol` | `center_neutral`, `left`, and `right` rows. |
| `no_ideology` | Only `left` and `right` rows. |
| `no_left` | Only `left` rows. |
| `no_right` | Only `right` rows. |
| `lr_balanced` | Extra `left` rows so the remaining left count matches the right count. |

Row lists are written under:

```text
data/interventions/dolci_olmo3_sft/
```

## Stage 21: DOLCI Token Accounting

This stage measures how much of the DOLCI corpus is affected by each label and
each intervention. This matters because the `think` split has much longer rows
on average than the `instruct` split, so row percentages and token percentages
can tell different stories.

| File | Role |
| --- | --- |
| `code/compute_dolci_olmo3_token_stats_by_label.py` | Counts Llama-tokenized tokens for every DOLCI row and aggregates by split and 4-way label, treating rows absent from the candidate/classification file as `not_candidate`. |
| `code/compute_dolci_intervention_token_removal_stats.py` | Uses the intervention row lists to estimate removed rows and removed tokens for each intervention variant across `instruct` and `think`. |

The main outputs are:

```text
data/interventions/dolci_olmo3_sft/token_stats_by_label.json
data/interventions/dolci_olmo3_sft/intervention_token_removal_stats.json
```

## Stage 22: Build OLMo-3 DOLCI Intervention JSONLs

This stage streams through the DOLCI flattened JSONL files and writes actual
trainable intervention datasets while preserving separate `instruct` and `think`
outputs.

| File | Role |
| --- | --- |
| `code/build_dolci_olmo3_intervention_jsonl.py` | Given `--variant`, removes row IDs from the appropriate DOLCI split and writes filtered JSONL files plus a summary JSON. |

Supported variants:

```text
no_pol
no_ideology
no_left
no_right
lr_balanced
```

Outputs are written under:

```text
data/interventions/dolci_olmo3_sft/jsonl/<variant>/
```

## How the Pieces Fit Together

At the highest level, the full project flow is:

```text
1. Build political benchmarks
   -> Manifesto chunks and benchmark v1/v2
   -> external v3 binary benchmarks

2. Test political classifiers
   -> prompted LLM classifiers
   -> encoder classifiers
   -> benchmark summaries and error analyses

3. Move into SFT data
   -> inspect Tulu/OLMo mixture
   -> flatten conversations
   -> keyword-filter candidate political examples

4. Produce manual truth
   -> 4-way classifier sample
   -> Streamlit/manual labels
   -> evaluate classifier errors

5. Train target-domain gate
   -> manual labels become binary training data
   -> ModernBERT gate predicts political/non-political

6. Test two-stage behavior
   -> ModernBERT gate first
   -> LLM 4-way classifier second
   -> evaluate on in-domain and OOD manual samples

7. Intervene on Tulu/OLMo-2 SFT data
   -> remove or rebalance detected political-position examples
   -> train OLMo-2 LoRA variants

8. Extend to OLMo-3 DOLCI SFT
   -> inspect and flatten Dolci-Instruct-SFT-7B and Dolci-Think-SFT-7B
   -> apply combined ModernBERT gate to both splits
   -> classify candidates with the 4-way LLM prompt
   -> compute label/token statistics by instruct vs think split
   -> build DOLCI intervention row lists and JSONL variants
```

## Recommended Reading Order

If you want to understand the codebase from scratch, read in this order:

1. `code/benchmark_utils.py`
2. `code/benchmark_prompts.py`
3. `code/build_benchmark_v1.py`
4. `code/run_benchmark_v1.py`
5. `code/evaluate_benchmark_v1.py`
6. `code/flatten_tulu_olmo2_sft.py`
7. `code/prefilter_tulu_olmo2_political_candidates.py`
8. `code/run_sft_political_4way_classifier.py`
9. `apps/manual_sft_labeler.py`
10. `code/create_target_domain_binary_train.py`
11. `code/train_encoder_target_domain_binary.py`
12. `code/apply_target_domain_binary_modernbert.py`
13. `code/evaluate_modernbert_on_ood_sample.py`
14. `code/evaluate_two_stage_ood.py`
15. `code/create_sft_intervention_jsonl.py`
16. `code/train_olmo2_lora_sft_variant01_100k.py`
17. `code/download_inspect_olmo3_dolci_sft.py`
18. `code/flatten_olmo3_dolci_sft.py`
19. `code/apply_modernbert_to_dolci_olmo3_sft.py`
20. `code/prepare_dolci_candidates_for_4way.py`
21. `code/create_dolci_olmo3_intervention_row_lists.py`
22. `code/compute_dolci_olmo3_token_stats_by_label.py`
23. `code/compute_dolci_intervention_token_removal_stats.py`
24. `code/build_dolci_olmo3_intervention_jsonl.py`

That reading path follows the intellectual arc of the project: define the
classification problem, build measurement, inspect SFT data, label examples,
train the gate, evaluate OOD behavior, then create intervention data. The final
DOLCI scripts show how the same idea is scaled to OLMo-3 `instruct` and `think`
SFT data with split-aware token accounting.

## Current Research Interpretation

The project is not merely classifying politics for its own sake. It is trying to
answer a causal data question:

> What happens if political-position examples are removed or rebalanced in SFT
> training data?

To answer that, the pipeline needs reliable detection. The current two-stage
classifier is a practical compromise:

- ModernBERT is cheap and fast, but may miss neutral political examples.
- The LLM 4-way classifier is more expressive, but expensive to run everywhere.
- Manual labels provide the audit trail for both.

The OOD work is especially important because it tests whether the gate learned a
real concept of political content or just memorized patterns from the first
manual sample.

The OLMo-3 DOLCI branch is the current scale-up of that idea. It keeps the
`instruct` and `think` SFT splits separate because their row lengths and training
roles differ. The `think` split contains much longer reasoning-style examples,
so token-level accounting is necessary: row counts alone can understate or
overstate the amount of training signal removed by an intervention.


---

# Appendix 10: Official-model IFEval comparison; retained pending status

Source: `validation/olmo3_sft_reproduction/reports/cpu_verified_comparison.md`.

# OLMo 3 Full SFT instruction-following validation

## Current results

| Evaluation | Strict prompt | Strict instruction | Loose prompt | Loose instruction |
|---|---:|---:|---:|---:|
| Full SFT (saved, matched-local) | 56.56% | 66.79% | 60.63% | 70.02% |
| Official final checkpoint (matched-local) | pending | pending | pending | pending |

The saved Full SFT result is valid for the matched-local comparison: all 541 prompts and 834 constraints use the pinned Google Research scorer. The official checkpoint result remains pending until its weights are downloaded with permission and GPU inference is run.

## Published OLMo 3 results

| Official 7B stage | IFEval (paper metric) | IFBench |
|---|---:|---:|
| Instruct SFT | 81.7 | 27.4 |
| Instruct DPO | 82.0 | 29.3 |
| Final Instruct (RLVR) | 85.6 | 32.3 |

The paper's IFEval value is prompt-level loose accuracy. It uses zero-shot chat formatting, temperature 0.6, top-p 0.95, a 32,768-token output cap, and averages three runs. The published Full SFT-comparable reference is therefore 81.7 for `allenai/Olmo-3-7B-Instruct-SFT`; the final official model reference is 85.6 for `allenai/Olmo-3-7B-Instruct`.

## Method differences

The existing Full SFT run is a single greedy run with a 1,280-token cap and the literal training wrapper `USER: ... ASSISTANT:`. The official checkpoints require their released chat template, including its default system message. Using each model's native input format avoids an input-format handicap, while keeping benchmark rows and scoring identical.

The numerical gap between saved Full SFT loose-prompt accuracy (60.63%) and the paper's SFT value (81.7%) is -21.07 percentage points, but it is protocol-confounded and is not a direct performance estimate. The matched-local official run supports a controlled checkpoint comparison. Three paper-protocol runs support comparison with the published values.

## Sources

- https://arxiv.org/pdf/2512.13961 (v2, 2026-04-14), Tables 24 and 26, Appendix A.8.1
- https://huggingface.co/allenai/Olmo-3-7B-Instruct
- ifeval::olmo3:adapt; primary metric prompt_level_loose_acc
