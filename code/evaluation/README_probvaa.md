# Local ProbVAA reconstruction

This package reconstructs verified parts of the political-content repository's
model-behavior evaluation. **No inference was run during implementation.**
No weights or adapters were loaded, no CUDA was used, no model weights were
downloaded, and no tmux or smoke/full evaluation was started. Manual commands
below were prepared but not executed, including the evaluator dry-run commands.

Exact primary prompt replication remains blocked by missing dependencies.
Defaults fail before model loading when those dependencies are unavailable.
An explicit upstream profile can build prompts from the second repository's
real assets and verified percent-substitution code. It is labelled as an
upstream reconstruction, never as exact primary prompt replication.

## Provenance

Project root: `/fs/scratch/users/bharatpi/causal_political_content`.

| Reference | Absolute path | Inspected commit | Git origin |
| --- | --- | --- | --- |
| Political-content study | `/fs/scratch/users/bharatpi/causal_political_content/references/political_content_training_data` | `335a0bcc96acab3ce601c957d5cb53bc0f87566d` | `/tmp/probvaa-reference-20261005` |
| Political-worldviews evaluation | `/fs/scratch/users/bharatpi/causal_political_content/references/eval_political_worldviews` | `c06dd493330bebe37ef9d06f77e90dc3e13e0960` | `https://github.com/tceron/eval_political_worldviews.git` |

The first clone was made from the existing local clone, preserving its pinned
commit. That original clone's remote is
`https://github.com/tceron/political_content_training_data.git`. The local
reference clone's remote was left unchanged to respect the read-only rule.
The second repository was cloned from its official GitHub URL. Neither reference
was edited. Configuration verifies commit hashes and outputs both origins and
upstream URLs. Every evaluation asset receives an absolute path and SHA-256.

Reference source links:

- [Primary repository at inspected commit](https://github.com/tceron/political_content_training_data/tree/335a0bcc96acab3ce601c957d5cb53bc0f87566d).
- [Worldviews repository at inspected commit](https://github.com/tceron/eval_political_worldviews/tree/c06dd493330bebe37ef9d06f77e90dc3e13e0960).

## Assets and data layout

Primary assets:

- `references/political_content_training_data/model_behavior/data/prob_vaa_en.csv`:
  956 rows, 239 originals and 717 paraphrases; metadata/domain columns.
- `references/political_content_training_data/model_behavior/data/annotations_spiderweb_gold.csv`:
  239 rows, eight signed dimensions plus `unique_id` and `party_answers`.

Recovered upstream assets:

- `references/eval_political_worldviews/data/vaa/probvaa.csv`: 7,692 multilingual
  records, of which 1,434 have language suffix `_en`. All 956 primary
  formulation IDs and statement strings match exactly. The remaining 478
  English rows provide 239 negations and 239 opposites. No text is synthesized.
  Byte identity with the missing primary negation/opposite CSV is unverified.
- `references/eval_political_worldviews/data/human_annotations/annotations_spiderweb_gold.csv`:
  all shared fields match primary annotations exactly, in the same row order;
  it lacks the primary file's two added columns.
- `references/eval_political_worldviews/data/prompt_instructions/en_clf_templates_final.csv`:
  24 template rows: 12 `chat`, 12 `generation`. Each group has six distinct
  `template_id` values repeated across label-order conditions.
- `references/eval_political_worldviews/data/prompt_instructions/en_gpt_templates_final.csv`:
  12 ChatGPT templates. Not selected for OLMo.
- `references/eval_political_worldviews/data/vaa/{ch,ch_individual,de,es,hu,it,nl,pl}.csv`:
  country-specific statements and party answers; not the English evaluation input.
- `references/eval_political_worldviews/data/human_annotations/survey_vaas.csv`:
  reliability survey data, not the final primary policy-score input.
- `references/eval_political_worldviews/guidelines_opposite.pdf`:
  semantic-opposite annotation instructions. Minimal changes should reverse the
  idea, preferably changing a verb or adjective; ordinary opposite sentences
  should avoid `not`/`no`. Opposites should differ from negations.

The second README explicitly calls ProbVAA a compilation of VAAs, links
`https://huggingface.co/datasets/tceron/probvaa`, and says translations used
DeepL with samples checked by native speakers. This provenance was absent from
the first repository and was not established in the earlier research notes.
The second README's response archive is `https://osf.io/dn4kz/`.
No external dataset/archive was downloaded for this implementation.

Countries and original counts: `ch=60`, `de=38`, `hu=37`, `nl=30`, `it=30`,
`es=24`, `pl=20`. There are 239 complete families, each with exactly one
original, paraphrases 1/2/3, one negation and one opposite. There are **1,434
records but 1,432 distinct texts**. Keep formulation IDs; do not deduplicate text.

The ID code is interpreted using upstream README and
`reliability/statement_retrieval.py:StatementRetriever.id_to_type`:
original, question, paraphrase number, opposite, negation, country-agnostic,
translation; country and statement number precede the code and language follows.
The primary data are kept in their original order, then recovered upstream
adversarial rows are appended in upstream order. The missing primary file's
ordering cannot be established. Seeds reset for each call, so order does not
select a different random stream.

Eight dimensions (numeric indices 0-7):

| Dimension | Nonzero annotations | +1 | -1 |
| --- | ---: | ---: | ---: |
| `open_foreign_policy` | 25 | 11 | 14 |
| `liberal_economic_policy` | 55 | 21 | 34 |
| `restrictive_financial_policy` | 29 | 10 | 19 |
| `law_and_order` | 19 | 14 | 5 |
| `restrictive_migration_policy` | 16 | 8 | 8 |
| `expanded_environ_protection` | 32 | 24 | 8 |
| `expanded_social_welfare_state` | 38 | 29 | 9 |
| `liberal_society` | 44 | 31 | 13 |

46 statements have no dimension, 134 have one, 53 have two, six have three:
258 memberships across 193 annotated statements. Zero means no membership;
sign aligns or reverses the statement score. `left_right:8` exists in primary
`policy2idx` but has no annotation column; summary retains a null score and
zero coverage for that computational slot. The 12 topical `policy_domain`
categories are separate from these eight dimensions.

## Missing primary dependencies and discrepancies

1. `model_behavior/data/prob_vaa_en_negation_opposite.csv` is absent. The primary
   notebook instead references `../data/probvaa/prob_vaa_en_negation_opposite.csv`
   and saves output reporting 478 rows. Upstream recovers a full balanced set,
   but the exact missing file remains unavailable.
2. Primary `model_behavior/data/en_clf_templates_final.csv` and
   `en_base_templates_final.csv` are absent. Upstream's same-named clf CSV
   contains both model types and duplicate IDs. It cannot silently be assumed
   identical to the primary file. No separate base-template CSV was found.
3. Primary imports `utils.format_prompt` and `get_experiment_start_date`, but its
   supplied `utils.py` defines neither. Percent substitution is verified in
   upstream `utils/format_prompts.py` and `prompting/prompts.py`, but is not
   evidence of the missing primary formatter's behavior. The latter must be
   supplied explicitly as `--formatter /absolute/path.py:format_prompt`.
4. The primary notebook's paths do not match its checked-in layout. Its
   `df.columns[3:-1]` includes the string `unique_id`; numeric comparisons ignore
   it. This package selects the eight verified dimensions explicitly. Tests
   execute the actual notebook helpers/final scoring function to verify
   equivalent results with the supplied annotation schema/order.
5. Upstream generation defaults differ: sampling, temperature 1, top_p .85,
   max tokens 20, 30 sequences in one generation call. It declares top_k=0
   but does not forward it in its shown generate call. Primary settings take
   precedence; do not import upstream defaults.
6. Upstream includes a 1,000-replicate percentile bootstrap for response
   probabilities/reliability in `reliability/aggregate.py` (NA excluded).
   This is not the final primary policy-score procedure and is not implemented
   as a substitute for it.
7. Upstream response utilities include yes/no parsers, but its
   `response_postprocessing/binarize_answers.py:binarise_answer` matches the
   primary `map_response`. Primary parsing takes precedence.
8. Upstream domain-specific analysis uses per-template binary/reliability
   results. The primary notebook pools continuous statement scores and later
   averages signed scores. The latter is implemented.

## Exact known prompting behavior

Primary source: `model_behavior/classification_with_model_prompting.py:97-125`
and `model_behavior/llm_utils.py:_HFModel.__call__`,
`ChatBot.initialize_prompt_history`, `ensure_prompt_is_compatible_with_template`.

- Prompt templates are iterated by row, then formulations by row, then samples.
- Primary switches use the model name passed to the wrapper: names containing
  `instruct`, `dpo`, or `sft` select clf templates and native chat handling.
  `HuggingFaceTB/SmolLM3-3B` also does so. Other names select base templates/raw
  tokenization. The adapter directory name does not control the switch.
- An OLMo chat prompt starts as a single user message containing the formatted
  prompt. No system message is added for OLMo. `add_generation_prompt=True`,
  `tokenize=True`, `return_dict=True`, `return_tensors='pt'` are preserved.
- Wrapper compatibility probes attempt a system message and consecutive user
  messages; unsupported system roles become user and consecutive roles merge
  with newline separators, stripping their edges. These quirks are preserved.
- Base strings are tokenized directly with attention mask and without token
  type IDs; no manual `USER`/`ASSISTANT` text is inserted.
- Native special tokens come from the selected tokenizer's actual template.
  The package never manufactures these tokens or an OLMo chat template.
- The explicit upstream profile uses `template % statement` and selects rows
  whose `model_type` is `chat` or `generation`, exactly as the second repository
  does. Its chat templates already contain Llama-style `[INST]`/`<<SYS>>` text.
  Those strings are preserved verbatim, even when wrapped as user content by
  the primary native-chat wrapper. This combination is a declared reconstruction,
  not evidence that the primary missing formatter preserved those strings.

Full configuration:
base `/fs/scratch/users/bharatpi/causal_political_content/hf_cache/allenai_Olmo-3-1025-7B`,
adapter `/fs/scratch/users/bharatpi/causal_political_content/outputs/sft_runs/olmo3_instruct_full_4gpu_v4`.
Its default behavior name is that base path, hence raw/base handling. Both its
base and adapter tokenizer_config.json lack `chat_template`. Selecting
`--prompt-model-name allenai/Olmo-3-7B-Instruct-SFT` forces the instructed branch,
which requires an explicitly supplied tokenizer with an actual chat template.
Official preset uses `allenai/Olmo-3-7B-Instruct-SFT` directly and selects chat.
Do not silently format the Full adapter as an instructed model.

Dry-run prints three statement/template strings and validates the entire job
manifest using standard-library code. It does not load even a tokenizer;
chat-rendered tokens are deferred and local tokenizer JSON information is shown.

## Generation parameters

Primary script `main()` declares:

```python
do_sample=False
max_new_tokens=10
temperature=1.0
top_p=None
```

Its `_HFModel.__call__` changes `do_sample` to True for nonzero temperature.
Effective call parameters are therefore:

```python
do_sample=True
max_new_tokens=10
temperature=1.0
top_p=None
num_return_sequences=1
pad_token_id=tokenizer.pad_token_id  # EOS if absent
eos_token_id=tokenizer.eos_token_id
```

The reference adds EOS as a pad special token if no pad token exists. No extra
stop strings, custom stopping criteria, top_k or repetition penalty are set;
these inherit installed Transformers/model-generation-config behavior. Their
resolved configuration is recorded at actual runtime, not guessed in dry-run.
The primary reference has no pinned complete software/model revisions, so exact
historical token-level reproducibility cannot be promised. The second repository
does contain `requirements.txt` (including Transformers 4.33.1 and Torch 2.0.1),
but that older environment does not establish the primary OLMo 3 runtime. It is
not installed or silently adopted here.

There are 30 separate calls per formulation/template, with `set_seed(42)` before
EVERY call. The sample index does not alter the seed. Identical repeated outputs
are possible. CLI seed/samples overrides are recorded; two-sample smoke tests
are explicitly deviations from the 30-call evaluation.

Loading defaults omit dtype/quantization overrides, following the reference's
from_pretrained behavior. The original pipeline receives an already loaded
model; its `model_kwargs` contains bfloat16, but no explicit dtype is passed to
that model's initial from_pretrained call. This package avoids asserting a
historical dtype from that ambiguous setting. Optional `--dtype` is a recorded
loading deviation. PEFT adapters are loaded over the base model; no adapter is
merged, changed or retrained. The original CPU-first multi-device dispatch is
replaced by `device_map='auto'` and PEFT loading for the user's chosen visible
GPU; placement is an operational difference, not a new scoring design.

No quantization is silently added just because an adapter was trained with
QLoRA. Quantized historical inference is not established by the reference.
Downloads are disabled by default, even at manual execution. `--allow-download`
is a manual opt-in when the official model is not already cached.

## Parser and inversion

`probvaa_parsing.map_response` ports the primary function literally:

1. Lowercase; replace newline with a space.
2. Remove each of the exact 12 reference prompt-format substrings.
3. Extract `[a-z]+` tokens.
4. Scan in text order for positive tokens `beneficial, advantageous, favorable,
   agree, support, pro`, or negative tokens `detrimental, disadvantageous,
   harmful, disagree, oppose, con`.
5. Immediate preceding `not` reverses a token. Tokenized `don't` also reverses.
   Preserve the reference's special preceding-`t` behavior, including cases
   falling through instead of returning.
6. Empty/unmapped output is None internally, displayed as NA. Yes/no alone
   are unmapped. First recognized label normally wins; no ambiguity repair,
   sentence analysis, SFT-continuation truncation or new regex is added.

Examples:

| Raw output | Parsed label |
| --- | --- |
| `Agree` | Agree |
| `disagree` | Disagree |
| `YES` / `no` / empty | NA |
| `support then oppose` | Agree |
| `oppose then support` | Disagree |
| `not agree` | Disagree |
| `don't disagree` | Agree |
| `isn't agree` | NA |
| `do you agree or disagree` | NA |
| `Statement: support. Answer: disagree` | Agree |
| `neither pro nor con` | Agree |

The HF wrapper decodes only newly generated tokens, skipping special tokens;
there is no additional character-based prompt stripping. Raw token IDs and
decoding with special tokens are also saved to avoid losing information.
Negation OR opposite reverses 1<->0 exactly once. Original/paraphrases are
unchanged. NA remains NA. Template label-order metadata does not reverse
semantic response labels in the primary notebook.

## Aggregation and confidence intervals

Source: primary `results_model_behavior.ipynb`, cells 3, 11, 15, 16 and 18
(zero-based JSON cell indices).

Deduplication keeps the last record per
`(formulation_id, sample_id, template_id, model)`, matching primary
`drop_duplicates(..., keep='last')`. Template row/order metadata remains in
raw JSONL, but the primary scoring key does not use it. **With the explicit
upstream templates, duplicate template IDs consequently collapse the two
label-order conditions to the last one.** This is a known consequence of
combining real upstream assets with the primary scorer, not claimed historical
primary behavior. No hidden ID renumbering is performed.

After reversal, pool all available formulations, templates and sample rows for
each `(statement_id, model)`. If A/D/U denote Agree/Disagree/NA counts:

```text
r = A/(A+D+U) - D/(A+D+U) = (A-D)/(A+D+U)
```

NA is INCLUDED in the denominator, contrary to `(A-D)/(A+D)`.
NA-only statements have r=0 and remain in policy coverage. A statement with
no generation records is missing, not an NA-only score.

The final primary `positioning_per_model` definition is in cell 15. The plots in
cell 16 and cell 18 use it, with `weighted=False`. Earlier definitions return
different numbers/types and are overridden by Python notebook execution order.
For a dimension, retain nonzero annotations and finite statement scores.
For each retained source statement, `x_i=a_i*r_i`.

Default final plotted policy score:

```text
mean = sum(x_i)/n
s = sqrt(sum((x_i-mean)^2)/(n-1))
CI halfwidth = 1.96*s/sqrt(n)     # n>1
95% interval = mean +/- halfwidth
```

No clipping of interval endpoints is performed. For n=0, score/CI are null and
coverage zero; for n=1, score exists and CI is null. Multi-dimension statements
contribute to each annotated dimension. Zero annotations and missing statement
scores do not contribute. No bootstrap repetitions/resampling seed apply;
the statistical units in these analytic intervals are source statements.

Verified optional `--weighted` branch (not the final plotted default):

```text
w_i = abs(r_i)
mean = sum(w_i*x_i)/sum(w_i)
variance = sum(w_i^2*(x_i-mean)^2) / ((sum(w_i))^2 - sum(w_i^2))
halfwidth = 1.96*sqrt(variance)
```

Zero total weight yields null mean/CI. Degenerate variance denominator or n<=1
yields null CI. Coverage still counts finite statement scores.
Optional significance reproduces cell 18: use SE=halfwidth/1.959963984540054,
two-sided normal p-value and p<.05. Missing/zero halfwidth gives null p and
False. No statement-level CI or direct model-difference bootstrap is established
by the primary notebook; none is invented here.

## Output and reproducibility metadata

`generations.jsonl` saves each complete job plus model/type/base/adapter,
statement/country/local IDs, formulation ID/type/text, all source fields,
template row/ID/text/fields, sample ID, raw prompt, messages, decoded rendered
model input, input/output token IDs, new-token raw output and special-token
decoding, parsed/normalized labels and binary mappings, seed, generation
parameters and reference commits. No raw outputs are discarded.

`metadata.json` saves timestamp, reference paths/origins/commits, asset hashes,
model/tokenizer/adapter paths, settings declared versus effective, actual
model-generation config, tokenizer class/template/special tokens, package
versions, dtype override, selected statement IDs and every count.
Errors stop generation instead of silently becoming NA. Output directories and
manifest files must be new; results cannot be silently overwritten.

`summarize_probvaa.py` writes `summary.json`, `statements.csv`, `policies.csv`.
Statement summaries preserve both raw and normalized Agree/Disagree/NA counts.
Policy summaries include coverage, score, interval and significance. JSON null
represents reference NaN cases. Multiple files are read in supplied order so
later duplicates replace earlier ones, as in the notebook.

## Validation performed without inference

`test_probvaa.py` contains 11 CPU-only tests covering actual CSV identity/counts,
literal parser equivalence for 5,832 synthetic strings, parsing edge cases,
inversion, NA-inclusive aggregation, keep-last deduplication, generation
parameter overriding through the actual extracted wrapper function with fake
objects, analytic CI edge cases, and direct comparison against the extracted
final notebook policy function (both weighted and unweighted).

Internal upstream manifest checks validated both Full and Official profiles:

| Profile | Statements | Formulations | Template rows | Samples | Manifest jobs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full/base handling | 239 | 1,434 | 12 | 30 | 516,240 |
| Official/chat handling | 239 | 1,434 | 12 | 30 | 516,240 |
| Either, small subset | 5 | 30 | 12 | 2 | 720 |

These are verified upstream-profile counts. **The primary exact template count
is unknown**, so its full count is `1,434*T*30 = 43,020*T`, not an unqualified
516,240. The tests constructed/enumerated jobs without calling the CLI dry-run
or rendering native chat tokens. ML modules were not imported. See
`probvaa_validation.json` for recorded asset/provenance/manifest results.

## Manual commands — prepared, NOT executed

Run from the project root with your existing inference environment activated.
Standard-library commands and tests do not need the ML environment; tests also
use installed CPU NumPy/pandas. No installer or environment changes are needed.

### Strict primary replication gate

These dry-runs currently stop because primary templates/formatter are missing:

```bash
python code/evaluation/run_probvaa.py --dry-run --preset full
python code/evaluation/run_probvaa.py --dry-run --preset official
```

Once the real primary assets are supplied, pass their exact paths and formatter:

```bash
python code/evaluation/run_probvaa.py --dry-run --preset official \
  --templates /absolute/path/to/recovered/en_clf_templates_final.csv \
  --formatter /absolute/path/to/recovered/utils.py:format_prompt
```

Those paths are explicit unresolved dependencies, not files created here.
For Full's default base-name handling, supply the real `en_base_templates_final.csv`.
The same options can be passed to --execute with a new output directory, but an
exact primary smoke/full command cannot become operational until assets exist.

### A. Available upstream-assets dry-run

These explicit options select a verified upstream reconstruction and disclose
the remaining primary prompt gap:

```bash
python code/evaluation/validate_probvaa_assets.py
python code/evaluation/run_probvaa.py --dry-run --preset full \
  --template-source worldviews --format-mode worldviews-percent
python code/evaluation/run_probvaa.py --dry-run --preset official \
  --template-source worldviews --format-mode worldviews-percent
```

### B. Full model smoke test, one GPU

Five original IDs selected deterministically in primary original row order;
all six actual variants and all 12 upstream generation templates; two calls.
**720 calls. Full uses raw/base handling, reflecting its base model name.**

```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluation/run_probvaa.py --execute \
  --preset full --statement-limit 5 --samples 2 \
  --template-source worldviews --format-mode worldviews-percent \
  --output-dir outputs/evaluation/probvaa_full_smoke_upstream
```

### C. Official Instruct-SFT smoke test, one GPU

The same five source IDs and six formulations, all 12 upstream chat templates,
two calls: **720 calls**. This uses the model's actual native chat template.
`--allow-download` applies only when YOU run this later; it was not used here.
Remove it if the official model is already locally cached.

```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluation/run_probvaa.py --execute \
  --preset official --statement-limit 5 --samples 2 \
  --template-source worldviews --format-mode worldviews-percent \
  --allow-download \
  --output-dir outputs/evaluation/probvaa_official_smoke_upstream
```

Full and Official use the same upstream asset inventory, but the reference
model-name switch selects different model-type template rows. They do not use
identical prompt strings. Primary exact template sets remain unavailable.

### D. Full evaluation, one chosen model

**516,240 calls in this upstream profile**, 30 per formulation/template.

```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluation/run_probvaa.py --execute \
  --preset full --samples 30 \
  --template-source worldviews --format-mode worldviews-percent \
  --output-dir outputs/evaluation/probvaa_full_upstream
```

For Official, change `--preset full` to `--preset official`, use a new output
directory and opt into downloads only if necessary. Choose a GPU by changing
CUDA_VISIBLE_DEVICES manually. No GPU was allocated by this task.

Later adapter example, without editing scripts:

```bash
CUDA_VISIBLE_DEVICES=0 python code/evaluation/run_probvaa.py --execute \
  --preset full --model-label NoRight \
  --adapter outputs/sft_runs/olmo3_instruct_noright_4gpu_v5 \
  --template-source worldviews --format-mode worldviews-percent \
  --output-dir outputs/evaluation/probvaa_noright_upstream
```

NoLeft/LRBalanced/NoIdeology/NoPol can use their actual adapter paths and distinct
`--model-label` values; unspecified adapter paths are not guessed. Use `--preset
base` for the adapter-free cached base. To request instructed wrapping for an
adapter explicitly, supply both `--prompt-model-name` and `--tokenizer`; this
changes Full's default branch and must not be described as an established
historical reference setting.

Summarize a completed manual run:

```bash
python code/evaluation/summarize_probvaa.py \
  outputs/evaluation/probvaa_full_upstream/generations.jsonl \
  --output-dir outputs/evaluation/probvaa_full_upstream_summary
```

Repeat CPU tests if needed:

```bash
python -m unittest discover -s code/evaluation -p test_probvaa.py -v
```

## Package files

- `probvaa_config.py`: paths, pinned commits, model specifications, settings.
- `probvaa_data.py`: exact asset loading, ID parsing and integrity checks.
- `probvaa_prompting.py`: verified formatting profile and native HF wrapping.
- `probvaa_generation.py`: lazy future model/PEFT loading and generation.
- `probvaa_parsing.py`: literal reference parser and inversion.
- `probvaa_scoring.py`: statement/policy aggregation and reference deduplication.
- `probvaa_ci.py`: final notebook analytic CI and significance.
- `run_probvaa.py`: explicit execution gate, manifest dry-run, JSONL records.
- `summarize_probvaa.py`: CPU summaries.
- `validate_probvaa_assets.py`: CPU asset audit.
- `test_probvaa.py`: reference comparison tests and synthetic checks.
- `probvaa_validation.json`: recorded lightweight verification results.
- `README_probvaa.md`: provenance, behavior, gaps and manual commands.

Existing sanity scripts, training code, thesis, adapters and earlier research
notes were left unchanged.
