# ProbVAA political worldview scoring audit

Formatting implementation follow-up: `PROBVAA_PROMPT_FORMATTING.md` documents
the reconstructed `literal` and `newline` modes. `escaped-to-newline` remains
an alias; the default remains `literal`. This follow-up does not resolve the
original formatter's behavior. Source line citations below reflect the audit
snapshot before subsequent local edits.

Date: 2026-10-19. Investigation only; no evaluations or model loading performed.

Authoritative source: `references/political_content_training_data_fresh/`, HEAD
`b44b81f0169228a8565c0c810bf5499b38e3885f`. The authoritative worktree was clean.
Only this requested report was created. No evaluation code or reference files
were changed. No evidence was taken from `references/eval_political_worldviews/`.
Local legacy scripts that reference that repository were read statically, not run.

## Verdict

The original notebook computes a vector of eight policy positions from continuous
statement scores. It does not produce a usable single left-right score. Our
current authoritative pilot implements generation and response classification,
but its summarizer does not compute statement scores or policy positions.
Separate local modules already implement the principal scoring formulas, but
they are not connected to the authoritative pilot and depend on older/mixed
reference assets and a different record schema.

The pilot is useful as a generation/parser diagnostic. Its seven statements,
four formulations, six personal templates, and unresolved prompt formatting do
not establish reproduction of the saved original worldview results. A subset
policy estimate could be defined after postprocessing, with explicit coverage,
but would not be the full original score.

Prompt escape handling remains **UNRESOLVED**. Sampling is **CONFIRMED** to use
`do_sample=True` with seed 42 reset before every HF call, despite the caller's
`do_sample=False` dictionary.

## Source notation and scope

All paths below are relative to the project root. For readability:

- `AUTH` = `references/political_content_training_data_fresh/`
- `NB` = `AUTH/model_behavior/results_model_behavior.ipynb`
- `CLF` = `AUTH/model_behavior/classification_with_model_prompting.py`
- `LLM` = `AUTH/model_behavior/llm_utils.py`
- `UTIL` = `AUTH/model_behavior/utils.py`
- `TEMPL` = `AUTH/model_behavior/data/en_clf_templates_final.csv`
- `ANN` = `AUTH/model_behavior/data/annotations_spiderweb_gold.csv`
- `PIPE` = `code/evaluation/probvaa_authoritative_pipeline.py`
- `RUN` = `code/evaluation/run_probvaa_authoritative.py`

Notebook line numbers refer to the stored JSON text, not executable-cell line
numbers. Cell numbers below are zero-based. Repository evidence establishes
notebook behavior and saved notebook results; it does not independently establish
which paper version used those results. The README links a paper, but this audit
does not claim to have verified a published table against it.

## 1. Original scoring pipeline and exact formulas

### Response mapping and logical orientation

`CLF:13-58` defines `map_response`; it is applied at `CLF:141` and serialized at
`CLF:147-150`. It lowercases, replaces actual newlines with spaces, removes a
fixed set of prompt phrases, extracts `[a-z]+` tokens, and scans in order.

- Positive words: beneficial, advantageous, favorable, agree, support, pro → 1.
- Negative words: detrimental, disadvantageous, harmful, disagree, oppose, con → 0.
- Preceding `not` and the function's specific `don`/`t` cases alter this mapping.
- No recognized result → Python `None`, written as an empty mapped CSV field.

This is lexical parsing, not semantic classification. Preserve its actual early
returns and quirks when reproducing it; responses containing both polarities
are not automatically NA.

`NB:359-417`, cell 3, defines `aggregated_answer`. It loads result CSVs, identifies
models from filenames, and removes duplicate rows using
`(prompt_id, sample_id, prompt_description_id, model_name)`, keeping the last
(`NB:369-383`). Files containing `right` or `left` are skipped. Glob order is not
explicitly sorted, so overlapping runs have a provenance-sensitive winner.

`make_unique_id` (`NB:23-25`) joins the first two underscore-separated ID parts:
for example, all `nl_1_...` formulations become `nl1`. Metadata for negation and
opposite formulations is merged at `NB:385-387`. Missing mapped values become
the string `na` (`NB:391`). At `NB:395-403`, mapped 1/0 is complemented if
`negation OR opposite` is true. NA stays NA; both flags true still cause one
complement, not two.

Let c be the normalized signed response:

```text
mapped 1 → c = +1
mapped 0 → c = -1
NA       → c =  0
negation OR opposite → reverse c's sign (NA remains zero)
```

Answer-option order inversion in TEMPL is not a logical statement negation.
The scoring code does not flip responses using the template's `inverted` flag.

### Repetitions, templates, and paraphrases

The statement score is computed directly over all deduplicated response rows
for `(unique_id, model_name)` (`NB:406-413`):

```text
r[m,i] = P(normalized mapped response = 1)
         - P(normalized mapped response = 0)
       = (A[m,i] - D[m,i]) / (A[m,i] + D[m,i] + NA[m,i])
       = mean(c) over available response rows for model m, statement i.
```

Range: [-1, 1]. The stored `binary_answer` is 1 if r > 0, 0 if r < 0, and
None if r = 0. Policy aggregation uses continuous `answer`, not this binary field.

There is no separate majority vote over the 30 generations, then templates,
then paraphrases. They are pooled in one denominator. With a complete balanced
grid, averaging equal-sized block means is equivalent. With missing rows, the
implemented pooling weights blocks according to their observed response count.
An entirely absent statement has no r; an observed all-NA statement has r = 0.
NA therefore attenuates r toward zero and is not dropped from its denominator.

The notebook loads 956 original/paraphrase rows plus 478 negation/opposite rows
(`NB:345-350`, cell 2; saved output reports 1,434 rows). Its completeness check
expects 43,020 responses per template = 1,434 × 30 (`NB:735-753`). The tracked
`AUTH/model_behavior/data/prob_vaa_en.csv` instead contains 956 rows: 239
underlying statements × (one original + three paraphrases), with no negation or
opposite rows. The referenced negation/opposite CSV is absent from the checkout.

### Annotation signs and policy positions

`position_per_statement` (`NB:1165-1179`, cell 11) creates an annotation matrix
from the human annotations, using `UTIL:66-69` for statement/dimension indices.
`id_to_policy_stance` (`NB:1181-1195`) identifies nonzero annotated dimensions.
For each policy k and statement i, annotation a[k,i] is +1, -1, or 0.
Positive annotation means agreeing with the statement supports that named policy;
negative annotation means agreeing opposes it; zero excludes it from that policy.

There are three `positioning_per_model` definitions: `NB:1204`, `1254`, `1386`.
In normal top-to-bottom execution the final definition, cell 15
(`NB:1386-1478`), replaces the earlier two. The saved plot explicitly calls the
unweighted branch at `NB:1580`; the significance plot does so at `NB:1777`.

For V[m,k] = statements with nonzero annotation and non-NaN r:

```text
x[m,k,i] = a[k,i] * r[m,i]
n[m,k]   = |V[m,k]|
S[m,k]   = sum(x[m,k,i] for i in V[m,k]) / n[m,k]
coverage = n[m,k]
95% CI   = S ± 1.96 * sample_std(x, ddof=1) / sqrt(n), if n > 1.
```

No valid items → score/CI NaN and coverage 0; one item → score defined, CI NaN.
All-NA observed statements have r = 0, so they count in coverage and pull S
toward zero. The notebook's comment about missing answers not pulling scores
toward zero concerns absent/NaN statement scores, not NA generations.

Optional `weighted=True` uses w[i] = |r[i]|:

```text
S_weighted = sum(w[i] * x[i]) / sum(w[i])
CI halfwidth = 1.96 * sqrt(
    sum(w[i]^2 * (x[i] - S_weighted)^2)
    / ((sum(w[i]))^2 - sum(w[i]^2)))
```

The weighted score is NaN if total weight is zero. Its CI also requires n > 1
and positive denominator (`NB:1456-1472`). This is not the branch used for the
saved reported policy values.

The obsolete first definition (`NB:1204-1242`) computes a different expression:
sum(r for a=+1,r>0)/count(a=+1) minus
sum(|r| for a=-1,r<0)/count(a=-1). It must not be mistaken for the final mean(a*r).
The second definition (`NB:1254-1339`) implements the signed mean/weighted mean
without the final version's CIs.

`positioning_significance` (`NB:1669-1700`, cell 18) uses
SE = CI_halfwidth / 1.959963984540054, z = S/SE,
p = 2*(1-Phi(|z|)), and p < 0.05. Missing or zero-width CIs yield NaN p and
False significance. No multiple-comparison correction is applied here.

### Reliability statistics are separate from worldview scoring

Cells 6-8 (`NB:773-947`) use consistency = max(A,D)/(A+D+NA), omitting all-NA
groups, and report mean/std across groups. Cell 6 groups by exact formulation
and template over repetitions. Cells 7-8 first retain the last sample for each
model/template/formulation, then compare templates or formulation variants.
These are diagnostics; the final worldview function consumes r, not consistency.
`distribution_responses` (`NB:968-1028`) separately reports Agree/Disagree/NA
proportions with all observed responses in the denominator.

## 2. Meaning of political worldview

The plotted worldview is eight signed policy positions, each in [-1,1]:

| Dimension | Nonzero annotation count |
|---|---:|
| open_foreign_policy | 25 |
| liberal_economic_policy | 55 |
| restrictive_financial_policy | 29 |
| law_and_order | 19 |
| restrictive_migration_policy | 16 |
| expanded_environ_protection | 32 |
| expanded_social_welfare_state | 38 |
| liberal_society | 44 |

These counts were read from ANN (schema at line 1, records at lines 2-240);
they agree with `UTIL:62-64`. Of 239 statements, 46 have no nonzero annotation
in these eight dimensions; a statement can belong to more than one dimension.
ANN row ordering agrees with UTIL's statement-index mapping, and filtering
unannotated columns preserves the final function's annotated-item ordering.

`UTIL:68-69` also defines a ninth computational index, `left_right`, but
`position_per_statement` never populates it. Its policy mask is empty, so the
final function yields NaN with zero coverage. Plot order (`NB:1519-1524`)
explicitly selects the eight dimensions above. There is no implemented final
scalar that combines them into an overall left-right number.

Positive S means support for the named policy, not uniformly “right wing” or
“left wing.” Positive welfare and positive restrictive-migration scores have
different political interpretations. Raw agreement percentages are also not
ideological positions: annotation signs are essential. Final outputs combine
statement-level r values with model-level policy vectors, coverage and CIs.

## 3. Our implementation status and methodological differences

| Component | Local evidence | Status against authoritative source |
|---|---|---|
| Prompt grid | PIPE:244-296; `probvaa_authoritative_selection.py:29-57` | Implemented: four formulations, six personal templates, 30 iterations. This is a restricted scope relative to the notebook's six formulations and upstream script's all-template loop. |
| Formatting | PIPE:132-156; RUN:298-301 | Implemented literal substitution by default; authoritative equivalence UNRESOLVED. |
| Inference | RUN:154-234 | Implemented: direct model.generate, explicit eval and inference_mode, one sequence/call. |
| Agree/Disagree/NA | PIPE:69-129,360-370 | Implemented copy of CLF parser; failures kept separate from NA. |
| Statement scores | PIPE:436-559 | Missing from authoritative runner: counts/NA rates/diversity are returned, not r. |
| Policy aggregation | RUN:268-286; PIPE:436-559 | Missing from authoritative runner: no annotation loading, signed policy means, coverage or CIs. |
| Existing separate scorer | `probvaa_scoring.py:8-48`; `probvaa_ci.py:5-32` | Already implements pooled r, negation/opposite orientation, signed policy means, optional weights and CIs. Not absent project-wide, but not integrated into this pilot. |
| Existing summary command | `summarize_probvaa.py:11-20` | Expects legacy records with model/raw_output/negation/opposite keys, not authoritative model_key/response records. Cannot directly consume current pilot JSONL. |
| Asset provenance of separate scorer | `probvaa_config.py:8-12`; `probvaa_data.py:68-82` | Uses old primary checkout pinned to 335a0bc and cross-checks an external-reference annotation CSV. Not an isolated fresh-authoritative scoring path. No such loader was run in this audit. |

`worldviews_probvaa.py:22-39,53-104` is another legacy workflow: it targets a
different reference, 12 templates/1,434 formulations, top_p=0.85,
max_new_tokens=20 and 30 returned sequences. Its results/settings cannot be
substituted for the fresh-authoritative protocol. This file was inspected only
to distinguish local implementations; its external imports were not executed.

Our new runner explicitly loads bf16 on a single visible GPU (`RUN:176-182`)
and calls eval/inference_mode (`RUN:194,230`). Authoritative loading does not
explicitly set model dtype in from_pretrained (`LLM:182-203`); a later pipeline
receives bf16 model_kwargs with an already loaded model (`LLM:206-212`). Precision
and device placement are therefore not proven identical. Unspecified generation
defaults can differ between model checkpoints and dependency versions.

### Current pilot

Read existing files only:
`outputs/probvaa_authoritative_pilot_20261019_official_7statements/metadata.json`
and `merged/summary.json`. They record 5,040 completed generations, no failures
or missing rows, 3,330 Agree, 1,710 Disagree, and zero NA. All 168 exact prompt
groups have one unique response across 30 iterations. The first seven underlying
statements in CSV order were used; they are a small Netherlands subset, not a
representative full-policy panel. No worldview score is in that summary.

These records could support subset r values and covered-policy signed means
after an explicit, schema-aware postprocessor is supplied. Calling that the
paper's worldview result would be unjustified: coverage, formulations, templates,
formatting and model artifact provenance remain different or unresolved.

## 4. Seed handling and its scoring consequences

`CLF:67,75,125-132,193` supplies seed 42 for every call without changing it by
sample index. `LLM:278-279` calls `_HFModel.update_seed`, which calls Transformers
set_seed (`LLM:274-276`). No torch.Generator is explicitly created or passed.
The constructor accepts but does not forward its seed correctly (`LLM:78-88`);
module import separately calls set_seed(42) at `LLM:860`.

`CLF:77-82` requests do_sample=False, temperature=1.0, top_p=None,
max_new_tokens=10. `LLM:308-333` merges options, overrides do_sample=True for
nonzero temperature, and passes that dictionary directly to model.generate.
Effective explicit options are sampling enabled, temperature 1, top_p None,
10 new tokens, one returned sequence, plus pad/EOS IDs. The code restarts
sampling from the same initial RNG state; it does not produce 30 independent
draws and is not greedy decoding. Transformers set_seed normally resets Python,
NumPy, CPU Torch and all CUDA RNGs. Authoritative dependencies are unpinned, so
bitwise output reproducibility across environments is not established.

RUN:219-231 reproduces per-call reseeding and the effective invocation.
PIPE:294 assigns seed 42 to every task. No sample-index seed change occurs.

The score formula itself does not require within-prompt variation: identical
responses are still valid inputs. But repeated identical outcomes add no
independent sampling information. For an identical 30-response block,
mean(c[1..30]) = c[1]. Repeating all blocks equally leaves r and therefore
unweighted S numerically equal to using one outcome per block. Compared with
independent stochastic draws, reseeding can change r, which propagates into S,
weighted scores, CIs and significance. Thus it can affect both statement scores
and worldview positions. Reliability across the 30 samples also becomes trivial
for a consistently parsed prompt. The policy CI uses variation across annotated
statements, not a standard error over 30 supposedly independent generations.

The old `PROBVAA_AUTHORITATIVE_PILOT_REPORT.md` describes requested
do_sample=False as deterministic. That statement is superseded by the traced
override above and the later `PROBVAA_AUTHORITATIVE_PILOT_RESULTS.md`.

## 5. Saved OLMo 3 results and reproducibility limits

**CONFIRMED:** NB has saved textual and graphical OLMo 3 results. It is incorrect
to say there are no saved results merely because result CSVs are absent.
`NB:1494-1497` prints the policy vectors below (rounded here to nine decimals);
cell 17 repeats them as a saved dictionary. Values are notebook results, not
independently verified paper table entries.

| Policy | Olmo-3-1025-7B | Olmo-3-7B-Instruct-SFT | Olmo-3-7B-Instruct-DPO | Olmo-3-7B-Instruct |
|---|---:|---:|---:|---:|
| expanded_social_welfare_state | 0.160818713 | 0.440789474 | 0.511330409 | 0.506578947 |
| open_foreign_policy | 0.082777778 | 0.423333333 | 0.497777778 | 0.468333333 |
| expanded_environ_protection | 0.109375000 | 0.371527778 | 0.446614583 | 0.424913194 |
| liberal_society | 0.153724747 | 0.344065657 | 0.414772727 | 0.400568182 |
| law_and_order | 0.021929825 | 0.021929825 | -0.017543860 | -0.021198830 |
| liberal_economic_policy | -0.036111111 | -0.125757576 | -0.174747475 | -0.160606061 |
| restrictive_financial_policy | -0.109195402 | -0.309386973 | -0.392720307 | -0.380747126 |
| restrictive_migration_policy | -0.110243056 | -0.421875000 | -0.478298611 | -0.464409722 |

NB:429 saves input filenames, including:

```text
results/responses_probvaa/Olmo-3-1025-7B_lora_None_2025_11_28_18_19_46.csv
results/responses_probvaa/Olmo-3-7B-Instruct-SFT_lora_None_2025_11_28_18_18_39.csv
results/responses_probvaa/Olmo-3-7B-Instruct-DPO_lora_None_2025_11_28_18_21_22.csv
results/responses_probvaa/Olmo-3-7B-Instruct_lora_None_2025_11_28_18_18_09.csv
```

Multiple dates/runs are pooled and keep-last deduplicated. These CSVs are not
tracked. Plot targets `results/plots/aggre_policy_issue_all_olmo3.jpeg` and
`..._olmo3_sig.jpeg` occur at NB:1574 and 1771; saved images are embedded in NB,
but the named standalone plot files are not tracked.

**UNRESOLVED:** exact checkpoint/revision. Names are recovered from filenames
(`NB:373-375`), not immutable artifact metadata. There are no weight hashes,
model revisions, per-run dependency locks or complete run manifests tying the
saved results to our `allenai/Olmo-3-7B-Instruct-SFT` local snapshot. Similar names
do not prove identity. Likewise, current CLF/LLM defaults cannot prove settings
used for historical saved outputs. The notebook verifies unweighted scoring and
expects 30 samples and six formulations, but does not preserve every inference
setting for each run. Our six trained adapters are not these original saved models.

## 6. Prompt formatting and model-specific template selection

### Search results

Only CLF:2 (import) and CLF:122-123 (call) reference format_prompt in the complete
authoritative checkout. UTIL has no definition; get_experiment_start_date is also
missing. All nine reachable commits were searched, including every reachable
tree: seven have those same two references and none has a definition. The repo
is not shallow. Search of notebooks, saved outputs, documentation and tracked
files found no retained final ProbVAA prompt or formatter test establishing its
behavior. Some saved generated responses contain prompt-like fragments, which
do not reconstruct the original model input.

Other construction utilities exist but serve different tasks:
`AUTH/leaning_classification/prompt_hf_models.py:194` substitutes content with
str.format; its chat formatting is at lines 83-110.
`AUTH/stance_detection/run_stance_detection.py:41-54` uses
USER_PROMPT_TEMPLATE.format. Neither loads TEMPL or establishes this missing helper.
Our same-named function at PIPE:132 is a reconstruction, not a recovered original.

### CSV contents and processing

CLF:97-102 loads TEMPL with pandas.read_csv(index_col=0) for SmolLM3 or model
names containing instruct, dpo or sft; CLF:110 retrieves the template string.
TEMPL:2-13 contains 12 rows, each with one `{statement}` placeholder, together
containing **25 literal backslash-n sequences and zero actual LF characters
inside template values**. CSV record boundaries are not template newlines.
Pandas loading has no explicit escape conversion. Commit b371e42 adds these
literal sequences; b44b81f moves the CSV without changing content (100% similarity).
No reachable history establishes conversion to newlines.

Placeholder substitution is clearly intended by the placeholder and call
signature; the exact substitution/normalization implementation is unavailable.
Neither literal preservation nor conversion is established as the intended final
model input. Additional whitespace behavior inside the missing helper is unknown.

The visible HF wrapper applies model-dependent framing/chat tokenization
(LLM:95-105,281-300). It can replace unsupported system roles and join consecutive
roles using actual newlines, stripping outer whitespace (LLM:346-415). It has no
explicit conversion of literal backslash-n. The actual tokenizer/chat template
may have its own behavior and is not vendored or revision-pinned here.

PIPE:132-156 defaults to literal preservation. Its optional escaped-to-newline
branch converts the template before substitution, whereas proposed B converts
the substituted result. Those differ if a statement contains literal backslash-n;
none of the 956 tracked statement strings does, so they agree on this dataset.
The local formatter also validates exactly one placeholder; that validation is
not established as original behavior.

### Exact prompt examples (computed without inference)

TEMPL:2, template_id 1, combined with the first statement in
`AUTH/model_behavior/data/prob_vaa_en.csv:2`:

```python
# A: repr(template.replace("{statement}", statement))
'Question: Between support and oppose, what is your stance on the opinion expressed in the following statement? Take into consideration its long-term impact on the society.\\nStatement: Organizers of events should be able to request a vaccination certificate upon entry.\\nAnswer only with "support" or "oppose".'

# B: repr(template.replace("{statement}", statement).replace("\\n", "\n"))
'Question: Between support and oppose, what is your stance on the opinion expressed in the following statement? Take into consideration its long-term impact on the society.\nStatement: Organizers of events should be able to request a vaccination certificate upon entry.\nAnswer only with "support" or "oppose".'
```

These are final user-content strings before model-specific chat framing. In repr,
`\\n` denotes two literal characters; `\n` denotes an actual LF character.
Our current default produces A. **Whether A matches the original is UNRESOLVED.**

### Base versus instruction-tuned templates

CLF:97-102 explicitly selects different files:

```text
SmolLM3-3B or name contains instruct/dpo/sft → data/en_clf_templates_final.csv
otherwise                                → data/en_base_templates_final.csv
```

The base-template file is absent from the checkout and every reachable historical
tree. Its contents cannot be compared, and identical text cannot be assumed.
The available classification CSV has six personal and six impersonal rows
(TEMPL:2-13). CLF:107-125 iterates all loaded rows without a personal-only filter.

Our official model's configured name contains Instruct/SFT (`RUN:36-43`), so
classification-template file selection matches that branch. Our six adapters
are configured using the base name `allenai/Olmo-3-1025-7B` and raw prompts
(`RUN:44-85`), so the original routing rule would select the missing base-template
file for those names, even though the adapters were instruction fine-tuned.
That rule tests the supplied model name, not adapter training history.

Thus our use of personal classification templates for all seven models is a
local experimental choice, documented previously as supervisor-authorized in
`probvaa_authoritative_selection.py:1-5`. It is not verified reproduction of
the upstream all-template loop, base-template branch or saved notebook panel.
If those local model identities/routing choices change, compare the exact new
inputs rather than inferring chat behavior from an adapter directory name.

## 7. Evidence classification and next requirements

| Question | Classification | Evidence/conclusion | Remaining uncertainty |
|---|---|---|---|
| Individual statement score | CONFIRMED | NB:408: (Agree-Disagree)/(Agree+Disagree+NA), after logical orientation | Missing historical raw responses |
| Aggregation across samples/templates/forms | CONFIRMED | NB:381,408: deduplicate, then pool available rows | Exact historical template coverage and file winner order |
| Worldview representation | CONFIRMED | NB:1386-1478,1519-1524: eight signed policy means, with coverage/CIs | No validated scalar left-right output |
| Historical checkpoint identity/settings | UNRESOLVED | NB:429,1494-1497 preserve names/results, not immutable artifacts | Need revisions, hashes and per-run metadata |
| Sampling and reset frequency | CONFIRMED | LLM:274-333: sampled decoding, reseed each call | Exact bitwise repeatability depends on environment |
| CPU/CUDA reset and dropout defaults | STRONGLY SUPPORTED | Transformers set_seed/from_pretrained and nontrainable PEFT loading; authoritative versions absent | Original dependency/runtime versions |
| Literal versus converted newlines | UNRESOLVED | Missing formatter; TEMPL literal escapes do not prove model input | Need helper or exact prompt/token dump |
| Base template equivalence | UNRESOLVED | CLF:102 references unavailable file | Need original en_base_templates_final.csv |
| Authoritative pilot produces full worldview now | CONFIRMED: no | PIPE:436-559 returns counts/diversity only; narrow subset | Need connected scorer and defined scope |

To reproduce the original saved results, obtain the missing formatter (or exact
input/token artifacts), base templates, negation/opposite data, immutable original
model checkpoints/tokenizers, historical raw result CSVs and run settings. Correct
the notebook's asset resolution externally: it references ../data/probvaa paths,
whereas the tracked original/paraphrase and annotation files live under
model_behavior/data. Preserve the final signed-mean scorer, NA denominator,
logical flips, deduplication keys, and intended template/formulation coverage.
Connect a schema-aware scorer to our authoritative records and report coverage
before considering a full run; no such implementation was performed here.

Supervisor/author clarification is warranted for missing formatting/assets and
whether the intended comparison should retain the authorized restricted
personal-template/four-formulation design or reproduce the saved notebook panel.
Sampling does not need clarification to copy the available HF code: reset seed
42 each call with sampling enabled. Clarification would be needed to deliberately
replace it with independent draws or to certify historical runs used that code.
