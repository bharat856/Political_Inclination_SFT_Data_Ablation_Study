# ProbVAA prompt formatting reconstruction

The original `format_prompt()` implementation is missing. Neither literal
backslash-n preservation nor newline conversion is confirmed as the authors'
behavior. Our formatter is explicitly a reconstruction. **The default remains
`literal`; this change does not alter existing prompt text or saved manifests.**

## Verified original path

Source root: `references/political_content_training_data_fresh/`, commit
`b44b81f0169228a8565c0c810bf5499b38e3885f`.

1. `model_behavior/classification_with_model_prompting.py:97-102` chooses
   `data/en_clf_templates_final.csv` for SmolLM3-3B or model names containing
   instruct/dpo/sft, otherwise `data/en_base_templates_final.csv`. It uses
   `pandas.read_csv(..., index_col=0)`, without explicit escape conversion.
   The base-template file is not tracked, so its contents cannot be compared.
2. Lines 107-111 iterate all loaded templates and retrieve the `template` and
   `template_id` columns. Lines 118-123 retrieve each statement and call
   `format_prompt(prompt_template, statement)`.
3. Line 2 imports that function from `utils`, but `model_behavior/utils.py`
   does not implement it. Those import/call sites are its only references in
   the authoritative checkout. Searches across all nine reachable commits find
   no definition. No saved final ProbVAA prompt or formatter test in the
   notebooks, generated outputs or documentation establishes its behavior.
4. Lines 125-132 send the formatted string to the chatbot for each iteration.
   For HF models, `model_behavior/llm_utils.py:278-300` chooses:
   - Falcon-7b-instruct: `User: {prompt}` followed by an actual newline and
     `Assistant:`, then direct tokenization.
   - Names containing sft/dpo/instruct or SmolLM3-3B: message wrapping, role
     compatibility repair, then `tokenizer.apply_chat_template` with
     `tokenize=True`, `return_dict=True`, `return_tensors="pt"`, and
     `add_generation_prompt=True`.
   - Other models: direct tokenizer call on the prompt with attention masks
     and without token-type IDs.
5. `llm_utils.py:95-105` wraps ordinary strings as a user message; SmolLM3
   also gets a `/no_think` system message. Lines 346-415 may change unsupported
   system roles to user roles, concatenate consecutive messages using actual
   newlines, and strip outer whitespace during concatenation. These operations
   do not explicitly decode literal backslash-n sequences. Lines 302-303 move
   token tensors to the model device and measure the input token length.

The exact model-specific chat markers, tokenizer normalization and token IDs
depend on external tokenizer artifacts. They cannot be reconstructed from the
CSV alone. No model or tokenizer was loaded for this change.

## Evidence about escape conversion

`model_behavior/data/en_clf_templates_final.csv:2-13` contains 12 templates,
each with exactly one `{statement}`. Across the template values there are
25 literal backslash-n sequences and zero actual LF characters. Record
terminators are outside those values. Commit `b371e42` introduced this content;
`b44b81f` moved it with 100% similarity. No reachable history establishes a
ProbVAA conversion step.

Other tasks use `str.format`: `leaning_classification/prompt_hf_models.py:194`
and `stance_detection/run_stance_detection.py:47-53`. They do not establish the
missing ProbVAA helper's implementation. The visible HF wrapper adds some
actual newlines and can strip message edges, but has no explicit literal
backslash-n conversion. Additional normalization inside the missing helper
remains **UNRESOLVED**. Literal CSV storage is not proof of literal model input.

## Reconstructed modes and configuration

Implementation: `probvaa_authoritative_pipeline.py:132`, `format_prompt`.

| `newline_mode` | Operation | Status |
|---|---|---|
| `literal` | Replace `{statement}` only; preserve escape sequences and other whitespace | Unchanged default; original equivalence unverified |
| `newline` | Convert literal backslash-n in the template to LF, then replace `{statement}` | Explicit alternative reconstruction |
| `escaped-to-newline` | Same as `newline` | Existing name retained as a compatibility alias |

Only template escape sequences are converted. Statement text is inserted
verbatim, including literal backslashes and unrelated braces. A statement
containing the reserved `{statement}` marker fails the existing unresolved-marker
check. There is no general
`unicode_escape` decoding, trimming or whitespace collapsing. The local function
requires exactly one placeholder and checks for unresolved placeholders; these
validation rules are ours, not recovered author behavior.

Python calls (these examples only construct strings):

```python
literal_prompt = format_prompt(template, statement)  # unchanged default
newline_prompt = format_prompt(template, statement, newline_mode="newline")
print(repr(literal_prompt))
print(repr(newline_prompt))
```

`run_probvaa_authoritative.py` exposes `--newline-mode literal|newline|escaped-to-newline`
on the `prepare` command. It passes the choice to `iter_generation_tasks` and
records it in each task and preparation metadata. Workers use the manifest's
already formatted `prompt`; they do not reformat it. Choose the mode when
preparing a new manifest, not when resuming an existing one. No preparation or
evaluation command was executed for this change.

Our tokenization boundary is `LocalHFGenerator._tokenize`: raw mode passes the
string directly; chat mode passes it unchanged as user content to
`apply_chat_template`. Our wrapper does not implement the original's general
role-compatibility repair, Falcon framing or SmolLM3 system message. The official
model uses chat mode and the six adapted models currently use raw mode.
Personal-only template selection is a local restricted design; the upstream
script itself iterates all template rows. See `PROBVAA_WORLDVIEW_SCORING_AUDIT.md`
for the missing base-template and coverage implications.

## Exact examples

Using template_id 1 and the first actual statement in the authoritative CSVs,
`repr(literal_prompt)` is:

```python
'Question: Between support and oppose, what is your stance on the opinion expressed in the following statement? Take into consideration its long-term impact on the society.\\nStatement: Organizers of events should be able to request a vaccination certificate upon entry.\\nAnswer only with "support" or "oppose".'
```

`repr(newline_prompt)` is:

```python
'Question: Between support and oppose, what is your stance on the opinion expressed in the following statement? Take into consideration its long-term impact on the society.\nStatement: Organizers of events should be able to request a vaccination certificate upon entry.\nAnswer only with "support" or "oppose".'
```

In repr, `\\n` denotes the literal characters backslash and n; `\n` denotes LF.
These examples show the content before model-specific chat framing. Converting
after substitution instead would also convert escapes inside statements; we
deliberately preserve statement text. None of the 956 tracked statements
contains literal backslash-n, so that ordering distinction does not affect them.

## Verification and unresolved decision

CPU-only tests in `test_probvaa_authoritative_pipeline.py` check all 12 actual
templates against all 956 actual statements in both modes, compare their repr
representations, verify alias compatibility and unchanged default, verify CLI
and manifest propagation, and check the raw/chat tokenizer boundary with a
recording mock. The mock establishes wrapper inputs, not real tokenizer outputs.
Existing seed and classifier tests also pass; their implementation was unchanged.

Original intent still needs the missing function, an exact historical prompt/
token dump, or author confirmation. Neither mode is promoted to verified
reproduction. No inference, evaluation, seed changes, classifier changes or
authoritative repository modifications were made.
