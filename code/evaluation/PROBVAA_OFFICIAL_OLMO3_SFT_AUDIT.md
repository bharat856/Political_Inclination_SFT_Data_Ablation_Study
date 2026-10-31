# ProbVAA audit for the official OLMo 3 Instruct-SFT checkpoint

Audit date: 2026-10-19

Scope: read-only inspection of `code/evaluation/worldviews_probvaa.py`, its tests and related configuration/runner modules, the local checkpoint at `validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT`, and the pinned source under `references/eval_political_worldviews/`. No GPU job or model-weight load was performed. The only repository change made by this audit is this report.

## Decision

**No-go for an official-checkpoint ProbVAA validation with `worldviews_probvaa.py` as currently written.** The runner is hard-wired to a base checkpoint plus one of six PEFT adapters, always selects the reference repository's base/generation templates, and never applies an instruction checkpoint's native chat template.

There is a second, modular path (`run_probvaa.py` plus `probvaa_generation.py`) that can load a standalone Hugging Face checkpoint and correctly invokes the downloaded tokenizer's native chat template. It is not a drop-in replacement for the pinned `worldviews_probvaa.py` protocol: it uses a different source protocol's generation settings, seed scope, prompt reconstruction, scoring, and output layout. It also overrides the official checkpoint's two-token EOS list with a single tokenizer EOS ID.

The official checkpoint should therefore be added as an explicit validation model to one selected protocol, with the protocol choice documented. Do not compare a run from the modular path with six runs from `worldviews_probvaa.py` without first making their prompts, sampling, seed behavior, parsing, and identifiers identical.

## What was verified

- The pinned reference checkout is `c06dd493330bebe37ef9d06f77e90dc3e13e0960`, as expected by both configuration paths.
- `worldviews_probvaa.build_jobs()` uses the unchanged reference `StatementRetriever` and `format_prompts` functions. It produces 17,208 prompt blocks: 1,434 English formulations × 12 template rows, with 30 generations per block (516,240 responses).
- The legacy generation constants match `references/eval_political_worldviews/prompting/generate_responses.py`: sampling enabled, temperature 1.0, top-p 0.85, 20 new tokens, 30 returned sequences, and seed 42 set once before loading/generation. The reference declares `top_k=0` as a function argument but does not pass it to `generate`; the replica correctly does not invent it.
- The local official tokenizer loads with Transformers 5.10.2 without loading model weights. It has a 2,638-character native chat template, pad ID 100277 and tokenizer EOS ID 100257. A one-user-message probe rendered a default system message, the user turn, and an assistant-generation prefix using `<|im_start|>`/`<|im_end|>` tokens.
- The checkpoint's `generation_config.json` declares EOS IDs `[100265, 100257]`, i.e. both `<|im_end|>` and `<|endoftext|>`.
- CPU tests passed:
  - `test_worldviews_probvaa.py`: 7/7.
  - `test_probvaa.py`: 11/11.
- Direct parser probes verified case and quote tolerance: `Agree`, `AGREE`, `"Agree"`, Unicode-quoted `“Agree”`, `Disagree`, and `"DISAGREE"` map as expected; `NA` and `N/A` map to no answer. When both labels occur, the first recognized answer label wins (`disagree or agree` → Disagree; `agree or disagree` → Agree), matching the released reference function.

Passing tests do not establish official-checkpoint readiness. The legacy suite never exercises a standalone checkpoint or the official tokenizer/chat template, and its parser fixture is almost entirely lowercase `agree`. The modular suite tests parser equivalence thoroughly but does not instantiate the official `Generator`, assert the rendered official prompt, or protect the checkpoint's EOS list.

## Findings by requested check

### 1. Standalone Hugging Face checkpoint support

**`worldviews_probvaa.py`: no.**

- `ADAPTERS` and `--model` admit only the six fine-tuned conditions.
- `configuration()` always opens an `adapter_config.json`, requires `adapter_model.safetensors`, and requires the fixed `BASE` checkpoint.
- `generate()` always imports PEFT, loads `BASE`, and calls `PeftModel.from_pretrained()`.
- The manifest schema requires adapter hashes and `compare_scores()` iterates exactly the six adapter names.

Files: `code/evaluation/worldviews_probvaa.py` (constants and `configuration`, `generate`, `main`, `compare_scores`); `code/evaluation/test_worldviews_probvaa.py` (only adapter-backed configurations are tested).

**Modular path: yes, with an explicit local override.** `model_spec('official')` has no adapter and `Generator` conditionally applies PEFT only when `adapter_path` is present. However, the preset itself names `allenai/Olmo-3-7B-Instruct-SFT`; because generation defaults to `local_files_only=True`, the already-downloaded directory is not automatically selected. A caller must currently provide `--base-model validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT` (and preferably the same explicit tokenizer path).

Files: `code/evaluation/probvaa_config.py`, `code/evaluation/run_probvaa.py`, `code/evaluation/probvaa_generation.py`.

Minimum change: introduce an explicit model specification in the chosen runner with `base_model`, optional `adapter_path`, `tokenizer_path`, display label, and chat/base behavior. Point the official validation entry at the existing local directory. Make PEFT loading and adapter provenance conditional.

### 2. Tokenizer and chat-template handling

**Legacy path: incorrect for the official instruction checkpoint.** It loads the fixed base tokenizer and calls `tokenizer(prompt)` directly. It neither loads the official tokenizer nor uses `apply_chat_template`.

**Modular path: mostly correct, with one material stopping bug.** It detects `Instruct-SFT`, loads that checkpoint's tokenizer, creates one user message, and calls `apply_chat_template(..., add_generation_prompt=True)`. The observed rendered input includes the checkpoint's built-in default system prompt. Runtime metadata records the full chat template, messages, rendered input, and input IDs.

The modular generator then sets `eos_token_id=self.tokenizer.eos_token_id` (100257), overriding the checkpoint generation configuration's `[100265, 100257]`. It may therefore continue after `<|im_end|>` until `<|endoftext|>` or the token cap. This can alter completions and classifications. Also, the dry-run helper reads only `tokenizer_config.json`; because this checkpoint stores its template in `chat_template.jinja`, dry-run metadata does not demonstrate that the template exists.

Files: `code/evaluation/worldviews_probvaa.py` (`generate`); `code/evaluation/probvaa_prompting.py` (`tokenize_prompt`, `local_tokenizer_info`); `code/evaluation/probvaa_generation.py` (`Generator.__init__`); checkpoint `chat_template.jinja`, `tokenizer_config.json`, and `generation_config.json`.

Minimum change: load tokenizer and model from the official local directory, render a single user message through the native template, and preserve the checkpoint's resolved EOS list rather than replacing it with the tokenizer's scalar EOS. Record and test the exact rendered prefix and token IDs. Extend dry-run inspection to fingerprint/report the separate Jinja template.

### 3. Original prompts, templates, sampling, and statement identifiers

**For the base/generation branch, `worldviews_probvaa.py` is faithful.** It uses the original CSV, unchanged formatter, all 1,434 English formulation IDs, the 12 `model_type=generation` rows, their family and final-template IDs, and the original generation constants.

**For an instruction checkpoint, its template selection is not faithful to the reference dispatcher.** The reference `generate_responses.py` selects `model_type=chat` when `llm_dict[model]['chat']` is true. Those 12 rows differ from the 12 generation rows. The chat rows also contain literal Llama-2 `<s>[INST]...[/INST]` serialization. Feeding those strings directly to OLMo is not correct OLMo serialization; wrapping them as the user content of the OLMo native template preserves the released strings but creates a documented model-specific reconstruction, not a literal reproduction of the original Llama run.

The modular path selects chat rows for the official model and preserves the exact CSV text, statement/formulation IDs, template family ID, row number, sample ID, and the complete template row. It then native-chat-wraps the literal Llama-marked string. Its generation protocol does **not** match the original worldviews code: 10 rather than 20 new tokens, no top-p 0.85, one returned sequence per call, and seed 42 reset before every call rather than one run-level RNG stream producing 30 sequences.

Files: `code/evaluation/worldviews_probvaa.py`; `code/evaluation/probvaa_config.py`; `code/evaluation/probvaa_prompting.py`; `references/eval_political_worldviews/prompting/generate_responses.py`, `utils/format_prompts.py`, `prompting/llm_dict.py`, and `data/prompt_instructions/en_clf_templates_final.csv`.

Minimum change: declare the exact target protocol before running. For validation of the pinned worldviews implementation, retain its sampling/seed behavior and select the reference chat rows for the official instruct checkpoint, while explicitly documenting the OLMo serialization adaptation. Preserve `statement_id`, `template_id`, `final_template_id`, CSV inversion flag, operational inversion flag, prompt index, and an explicit response index. Do not silently use the modular path's later-paper defaults.

### 4. Completion-only parsing

**Legacy path: conditionally correct, but unnecessarily fragile.** It decodes the full prompt plus completion, requires every decoded string to begin exactly with the raw prompt, and applies the reference character slice before parsing. It also computes and saves the token-sliced `generated_suffixes`, but does not parse those suffixes. The prefix assertion makes the current raw/base case safe; it will fail or become invalid once a native chat wrapper is introduced because the rendered model input is no longer equal to the raw prompt string.

**Modular path: correct.** It slices generated token IDs at the input-token length, decodes only that suffix into `raw_output`, and passes only `raw_output` to `map_response`.

Files: `code/evaluation/worldviews_probvaa.py` (`original_answer`, `generate`, `read_segments`); `code/evaluation/probvaa_generation.py` (`generate`); `code/evaluation/run_probvaa.py`.

Minimum change: make the legacy runner parse its already-computed token-sliced completion, not a character slice of full decoded text. Retain full/rendered input separately for auditability. Validate saved rows by reparsing the saved completion field.

### 5. Agree/Disagree/NA, capitalization, quotes, and reversed order

The classification function matches the released reference function. It lowercases text, extracts alphabetic words (therefore ignoring straight or curly quotes), maps recognized positive/negative synonyms, handles the implemented `not`/`don't` cases, and returns no answer for strings with no recognized label. `NA` is not a special literal; it becomes NA because neither `na` nor `n/a` is a recognized stance. This behavior is reference-compatible.

When a completion contains both answer labels, the first recognized label wins. That makes explicit answer ordering operational (`Disagree or Agree` maps Disagree), but it also means explanations or echoed alternatives can be classified by their first label rather than their final answer. That is an inherited reference limitation and should be reported, not silently "improved," for a reproduction.

Reversed prompt ordering is handled outside the parser. `worldviews_probvaa.py` reproduces the source postprocessor's positional rule: the first 30 responses for a template-family/statement key are operational inversion 0 and the next 30 are inversion 1, regardless of the CSV `inverted` value. Both the CSV flag and operational flag are retained. This matches the released code, including its ordering dependence. The tests only check that both values exist, not that each exact final-template row maps to the expected operational value.

Files: `references/eval_political_worldviews/response_postprocessing/binarize_answers.py`; `code/evaluation/worldviews_probvaa.py`; `code/evaluation/probvaa_parsing.py`; both ProbVAA test files.

Minimum change: do not change the reference parser for the validation run. Add explicit fixtures for case, straight/curly quotes, NA/N/A, both label orders, negation, and every reversed template row. Add a diagnostic field for multiple recognized labels so the paper-facing distribution can disclose ambiguous-but-mapped completions.

### 6. Reproducing the paper's response distribution

The legacy generation records contain enough response-level data to recompute a raw distribution: prompt and IDs, all 30 full outputs, token-sliced suffixes, output token IDs/probabilities, parsed integer or `not_applied`, generation settings, source hashes, runtime package versions, and RNG state. Scoring writes `parsed_responses.csv`, which includes the extracted answer and binary result.

It does not directly emit an Agree/Disagree/NA distribution, an ambiguity count, or a stable per-response identifier. More importantly for an official baseline, it has no official-checkpoint mode and its provenance schema hashes only the fixed base config/tokenizer and adapter files; it does not hash the standalone checkpoint index/shards, chat template, tokenizer support files, or generation configuration. Thus a future official run copied into this layout would not yet establish which exact checkpoint bytes produced the distribution.

The modular JSONL is more convenient for distribution reporting (`parsed_answer`, normalized label, raw output, messages, rendered input, input/output IDs, prompt/sample identifiers), and `summarize_probvaa.py` emits per-statement raw and normalized Agree/Disagree/NA counts. It still does not emit the overall distribution shown in a paper, and its metadata lacks checkpoint-weight hashes/revision. Its statement formula includes NA in the denominator; that differs from the original 2024 reliability bootstrap, which excludes `not_applied`. The intended updated-paper convention must be named explicitly in the report.

Files: `code/evaluation/worldviews_probvaa.py` (`configuration`, `generate`, `score`); `code/evaluation/run_probvaa.py`; `code/evaluation/summarize_probvaa.py`; `code/evaluation/probvaa_scoring.py`.

Minimum change: add immutable checkpoint fingerprints (index and all shards, config, generation config, tokenizer JSON/config, special-token map, merges/vocabulary, and chat template), explicit response IDs/indexes, and a derived distribution artifact with raw counts and denominators for Agree, Disagree, NA, multi-label/ambiguous, and cap-hit responses. State whether NA is included in the denominator and report both raw and formulation-normalized labels where relevant.

### 7. Non-overwriting pilot

The legacy runner has `--limit-prompts` and `--stop-after`; it refuses a nonempty directory without a matching manifest and opens new segment files exclusively. A brand-new output directory can therefore hold a pilot without overwriting prior results. It cannot currently run that pilot on the official checkpoint, and taking the first N prompt blocks is not a balanced diagnostic sample.

The modular runner provides a better isolation guard for a new pilot: `--statement-limit`, `--samples`, and `output.mkdir(..., exist_ok=False)`. Its optional manifest is also opened exclusively. It cannot overwrite an existing output directory, although a failed initialization can leave a newly-created partial directory. It has no in-place resume mechanism. Scientifically, its current official pilot would still carry the protocol and EOS problems above.

Files: `code/evaluation/worldviews_probvaa.py` (`generate`, CLI); `code/evaluation/run_probvaa.py`.

Minimum change: after the protocol fixes, use an explicitly new validation directory and a balanced, frozen pilot manifest rather than `jobs[:N]`. Include all six formulation types and both label orders in the pilot. Keep official-baseline outputs outside all six fine-tuned-model directories. Add an exclusive completion marker/summary so a partial directory cannot be mistaken for a complete pilot.

## Minimum implementation plan

1. Choose one canonical protocol and write its identity into metadata. For a direct validation of `worldviews_probvaa.py`, use the pinned worldviews data, formatter, 12 instruction/chat rows, 30 samples, temperature 1.0, top-p 0.85, 20-token cap, and run-level seed 42. Clearly label native OLMo chat serialization as the model-specific adaptation.
2. Refactor `worldviews_probvaa.py` model configuration to support either a standalone HF checkpoint or a base-plus-adapter checkpoint. Add the official local checkpoint as a validation-only model without adding it to the six-model comparison loop.
3. Load the official tokenizer, apply its native template, preserve both checkpoint EOS IDs, and record/fingerprint the rendered template and all checkpoint assets.
4. Parse token-sliced completions. Add explicit response indices and parser ambiguity/cap diagnostics while retaining the unchanged reference label result.
5. Add CPU tests for standalone loading configuration (mocked model load), the real local tokenizer/rendered prompt, EOS resolution, exact chat-template row selection/IDs, completion-only parsing, all classification edge cases, and exclusive pilot output behavior.
6. Add a CPU-only distribution summarizer that reports Agree/Disagree/NA and denominators without requiring the full political-scoring stage.
7. Only after those checks pass, run a small balanced pilot in a new directory. Do not launch the six fine-tuned models until the official baseline's rendered inputs, stopping behavior, classifications, and distribution artifact have been reviewed.

## Exact files requiring changes

Required for the recommended legacy-runner extension:

- `code/evaluation/worldviews_probvaa.py`
- `code/evaluation/test_worldviews_probvaa.py`

Likely new or extended configuration/reporting support:

- a small model-spec/config file under `code/evaluation/` (or an equivalent data structure in `worldviews_probvaa.py`)
- a CPU-only distribution summarizer and its test under `code/evaluation/`

If the modular runner is chosen instead, the minimum touched set becomes:

- `code/evaluation/probvaa_config.py`
- `code/evaluation/probvaa_generation.py`
- `code/evaluation/probvaa_prompting.py`
- `code/evaluation/run_probvaa.py`
- `code/evaluation/summarize_probvaa.py`
- `code/evaluation/test_probvaa.py`

No reference file under `references/eval_political_worldviews/` and no checkpoint file under `validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT/` should be modified.
