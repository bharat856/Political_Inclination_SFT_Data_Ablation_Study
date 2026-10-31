"""Run the pinned ProbVAA protocol with local OLMo 3 LoRAs or official SFT.

The reference repository is read only. This file supplies model loading, durable
generation records, and an output adapter for the reference postprocessors.
"""
from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "references/eval_political_worldviews"
REFERENCE_COMMIT = "c06dd493330bebe37ef9d06f77e90dc3e13e0960"
BASE = ROOT / "hf_cache/allenai_Olmo-3-1025-7B"
DATA = SOURCE / "data/vaa/probvaa.csv"
TEMPLATES = SOURCE / "data/prompt_instructions/en_clf_templates_final.csv"
ANNOTATIONS = SOURCE / "data/human_annotations/annotations_spiderweb_gold.csv"
ADAPTERS = {
    "Full": "olmo3_instruct_full_4gpu_v4",
    "LRBalanced": "olmo3_instruct_lrbalanced_4gpu_v5",
    "NoIdeology": "olmo3_instruct_noideology_4gpu_v5",
    "NoLeft": "olmo3_instruct_noleft_4gpu_v4",
    "NoRight": "olmo3_instruct_noright_4gpu_v5",
    "NoPolitical": "olmo3_instruct_nopol_4gpu_v5",
}
OFFICIAL_MODEL = "OfficialSFT"
OFFICIAL_CHECKPOINT = ROOT / "validation/olmo3_sft_reproduction/models/Olmo-3-7B-Instruct-SFT"
GENERATION = dict(do_sample=True, temperature=1.0, top_p=0.85,
                  max_new_tokens=20, num_return_sequences=30)
OFFICIAL_EOS_TOKEN_IDS = [100265, 100257]
POSITIVE_WORDS = {"beneficial", "advantageous", "favorable", "agree", "support", "pro"}
NEGATIVE_WORDS = {"detrimental", "disadvantageous", "harmful", "disagree", "oppose", "con"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_modules():
    # The original files are imported without copying or altering their functions.
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from reliability.statement_retrieval import StatementRetriever
    from utils.format_prompts import format_prompts
    from response_postprocessing.binarize_answers import (
        binarise_answer, binerarize_answers_two_sided,
    )
    return StatementRetriever, format_prompts, binarise_answer, binerarize_answers_two_sided


def build_jobs(model_type="generation"):
    import pandas as pd
    if model_type not in {"generation", "chat"}:
        raise ValueError(f"Unsupported reference template model_type: {model_type}")
    StatementRetriever, format_prompts, _, _ = reference_modules()
    statements = pd.read_csv(DATA)
    # The original get_english_statements() also returns translations to other
    # languages. The requested experiment is explicitly the six English forms.
    statements = statements[statements["ID"].str.endswith("_en")].copy()
    statements["statement_id"] = statements["ID"]
    english = StatementRetriever(statements).get_english_statements()
    templates = pd.read_csv(TEMPLATES)
    templates = templates[templates["model_type"] == model_type].copy()
    if len(english) != 1434 or len(templates) != 12:
        raise ValueError("Unexpected original English dataset/template coverage")
    if len(set(english["ID"])) != 1434 or len(set(templates["final_template_id"])) != 12:
        raise ValueError("Duplicate original formulation or final template ID")
    ids, statement_ids, prompts = format_prompts(templates, english, False)
    jobs = []
    seen_family = set()
    for template in templates.itertuples(index=False):
        final_id = int(template.final_template_id)
        family = int(template.template_id)
        for statement in english.itertuples(index=False):
            offset = len(jobs)
            if str(ids[offset]) != str(family) or statement_ids[offset] != statement.ID:
                raise ValueError("Original formatter ordering differs from source rows")
            # The reference binarizer calls the first 30 responses for a
            # family/statement non-inverted, regardless of the CSV flag.
            key = (family, statement.ID)
            operational_inverted = int(key in seen_family)
            seen_family.add(key)
            jobs.append(dict(prompt_index=offset, statement_id=statement.ID,
                             template_id=family, final_template_id=final_id,
                             csv_inverted=int(template.inverted),
                             operational_inverted=operational_inverted,
                             prompt=prompts[offset]))
    if len(jobs) != 17208 or len(seen_family) != 6 * 1434:
        raise ValueError("Original 12-template / 1,434-formulation grid incomplete")
    return jobs


def source_statement_id(statement_id):
    return "_".join(str(statement_id).split("_")[:2])


def balanced_pilot_jobs(jobs, statement_count):
    """Select whole source statements round-robin by country.

    Every selected source statement retains all six formulations, all six
    template families, and both answer-order rows (72 prompt blocks).
    """
    if statement_count < 1:
        raise ValueError("Balanced pilot statement count must be positive")
    by_country = {}
    seen_source_ids = set()
    for job in jobs:
        source_id = source_statement_id(job["statement_id"])
        if source_id not in seen_source_ids:
            seen_source_ids.add(source_id)
            country = source_id.split("_", 1)[0]
            by_country.setdefault(country, []).append(source_id)
    selected = []
    round_index = 0
    countries = sorted(by_country)
    while len(selected) < statement_count:
        added = False
        for country in countries:
            if round_index < len(by_country[country]):
                selected.append(by_country[country][round_index])
                added = True
                if len(selected) == statement_count:
                    break
        if not added:
            raise ValueError(f"Requested {statement_count} source statements, only {len(selected)} available")
        round_index += 1
    selected_set = set(selected)
    result = [job for job in jobs if source_statement_id(job["statement_id"]) in selected_set]
    for source_id in selected:
        group = [job for job in result if source_statement_id(job["statement_id"]) == source_id]
        if len(group) != 6 * 12:
            raise ValueError(f"Balanced pilot lost formulations/templates for {source_id}")
        if len({job["statement_id"] for job in group}) != 6:
            raise ValueError(f"Balanced pilot lost a formulation for {source_id}")
        for family in {job["template_id"] for job in group}:
            family_group = [job for job in group if job["template_id"] == family]
            if {job["operational_inverted"] for job in family_group} != {0, 1}:
                raise ValueError(f"Balanced pilot lost an answer order for {source_id}/{family}")
    return result, selected


def configuration(model, jobs, limit=None):
    import subprocess
    adapter = ROOT / "outputs/sft_runs" / ADAPTERS[model]
    ac = json.loads((adapter / "adapter_config.json").read_text())
    if Path(ac["base_model_name_or_path"]).resolve() != BASE.resolve():
        raise ValueError(f"{model} points to a different base model")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter / "adapter_model.safetensors")
    selected = jobs if limit is None else jobs[:limit]
    if not selected:
        raise ValueError("No prompts selected")
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError("Original worldviews repository revision changed")
    if subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain"], text=True).strip():
        raise ValueError("Original worldviews repository has local changes")
    source_files = [DATA, TEMPLATES, ANNOTATIONS,
                    SOURCE / "prompting/generate_responses.py",
                    SOURCE / "prompting/huggingface_llm_wrapper.py",
                    SOURCE / "utils/format_prompts.py",
                    SOURCE / "reliability/statement_retrieval.py",
                    SOURCE / "response_postprocessing/binarize_answers.py",
                    SOURCE / "reliability/aggregate.py",
                    SOURCE / "reliability/evaluation.py",
                    SOURCE / "reliability/utils.py",
                    SOURCE / "political_biases/domain_specific.py",
                    SOURCE / "political_biases/utils.py"]
    packed = json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()
    return dict(model=model, adapter=str(adapter.resolve()), base=str(BASE.resolve()),
                reference_commit=commit, source_sha256={str(p.resolve()): digest(p) for p in source_files},
                adapter_sha256=digest(adapter / "adapter_model.safetensors"),
                adapter_config_sha256=digest(adapter / "adapter_config.json"),
                base_config_sha256=digest(BASE / "config.json"),
                tokenizer_sha256=digest(BASE / "tokenizer.json"),
                prompt_manifest_sha256=hashlib.sha256(packed).hexdigest(),
                prompt_count=len(selected), full_prompt_count=len(jobs),
                generations_per_prompt=30, generation=GENERATION,
                seed=42, model_type="generation", language="en",
                prompt_format="original template % statement; no added wrapper",
                notes=["The original selector includes non-English translations; input is restricted to IDs ending _en.",
                       "The original postprocessor assigns the first 30 family responses inverted=0 and the next 30 inverted=1; the CSV inverted field is retained separately."])


def checkpoint_provenance(checkpoint=OFFICIAL_CHECKPOINT):
    checkpoint = Path(checkpoint).resolve()
    required = [
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "chat_template.jinja",
    ]
    missing = [name for name in required if not (checkpoint / name).is_file()]
    shards = sorted(checkpoint.glob("model-*.safetensors"))
    if missing or not shards:
        raise FileNotFoundError(f"Incomplete official checkpoint; missing={missing}, shards={len(shards)}")
    files = [checkpoint / name for name in required] + shards
    return {
        "path": str(checkpoint),
        "files": {path.name: {"size": path.stat().st_size, "sha256": digest(path)} for path in files},
    }


def official_configuration(jobs, selected_source_ids=None):
    import subprocess
    if not jobs:
        raise ValueError("No official-checkpoint prompts selected")
    commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if commit != REFERENCE_COMMIT:
        raise ValueError("Original worldviews repository revision changed")
    if subprocess.check_output(
            ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True).strip():
        raise ValueError("Original worldviews repository has local changes")
    source_files = [DATA, TEMPLATES, SOURCE / "prompting/generate_responses.py",
                    SOURCE / "prompting/huggingface_llm_wrapper.py",
                    SOURCE / "utils/format_prompts.py",
                    SOURCE / "reliability/statement_retrieval.py",
                    SOURCE / "response_postprocessing/binarize_answers.py"]
    packed = json.dumps(jobs, sort_keys=True, ensure_ascii=False).encode()
    return dict(
        model=OFFICIAL_MODEL, model_type="standalone_hf_checkpoint",
        checkpoint=checkpoint_provenance(), reference_commit=commit,
        source_sha256={str(path.resolve()): digest(path) for path in source_files},
        prompt_manifest_sha256=hashlib.sha256(packed).hexdigest(),
        prompt_count=len(jobs), full_prompt_count=12 * 1434,
        generations_per_prompt=30, generation=GENERATION,
        eos_token_ids=OFFICIAL_EOS_TOKEN_IDS, seed=42,
        seed_behavior="initialized once before model loading; one continuous run-level RNG stream",
        language="en", reference_template_model_type="chat",
        prompt_format=("exact reference chat-template CSV row formatted with % statement, then supplied "
                       "as one user message to the official tokenizer native chat template"),
        selected_source_statement_ids=selected_source_ids,
        notes=[
            "All six formulations and both answer-order rows are retained for every selected source statement.",
            "The reference chat rows contain literal Llama [INST] markers; retaining them inside the OLMo user message is a documented model-specific reconstruction.",
            "The official checkpoint is validation-only and is not part of the six-adapter comparison loop.",
        ])


def render_official_prompt(tokenizer, prompt):
    if not tokenizer.chat_template:
        raise ValueError("Official tokenizer has no chat template")
    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True)
    return tokens, messages


def official_generation_settings(tokenizer):
    checkpoint_generation = json.loads(
        (OFFICIAL_CHECKPOINT / "generation_config.json").read_text())
    if checkpoint_generation.get("eos_token_id") != OFFICIAL_EOS_TOKEN_IDS:
        raise ValueError("Official checkpoint EOS IDs changed")
    if tokenizer.pad_token_id is None:
        raise ValueError("Official tokenizer has no pad token ID")
    return GENERATION | {
        "eos_token_id": list(OFFICIAL_EOS_TOKEN_IDS),
        "pad_token_id": tokenizer.pad_token_id,
    }


def completion_diagnostic(completion, token_ids):
    """Diagnostics only; the returned classification still comes from the reference parser."""
    words = re.findall(r"[a-z]+", completion.lower())
    positive = [word for word in words if word in POSITIVE_WORDS]
    negative = [word for word in words if word in NEGATIVE_WORDS]
    eos_positions = [i for i, token_id in enumerate(token_ids)
                     if token_id in OFFICIAL_EOS_TOKEN_IDS]
    return dict(
        recognized_positive_words=positive,
        recognized_negative_words=negative,
        contains_both_label_polarities=bool(positive and negative),
        eos_token_id=(token_ids[eos_positions[0]] if eos_positions else None),
        hit_max_new_tokens=(not eos_positions and len(token_ids) >= GENERATION["max_new_tokens"]),
    )


def create_new_output_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def original_answer(full_text, prompt):
    # Exact character slice in the reference binarize_answers.py:81-84.
    return full_text[len(prompt) - 3:] if prompt.startswith("<s>") else full_text[len(prompt):]


def read_segments(directory, jobs, parser):
    records = []
    for segment in sorted(directory.glob("segment_*.jsonl")):
        with segment.open(encoding="utf-8") as handle:
            lines = handle.readlines()
        for line_no, line in enumerate(lines, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                if line_no == len(lines) and not line.endswith("\n"):
                    break  # Preserve a torn final line; resume in a new segment.
                raise ValueError(f"Corrupt completed record: {segment}:{line_no}") from error
            index = len(records)
            if index >= len(jobs) or row["prompt_index"] != index:
                raise ValueError("Saved prompts have a gap, overlap, or changed order")
            job = jobs[index]
            for field in ("prompt", "statement_id", "template_id", "final_template_id",
                          "csv_inverted", "operational_inverted"):
                if row[field] != job[field]:
                    raise ValueError(f"Saved {field} differs at prompt {index}")
            if len(row["outputs"]) != 30 or len(row["parsed"]) != 30:
                raise ValueError(f"Incomplete 30-response block at prompt {index}")
            if any(parser(original_answer(text, job["prompt"])) != (None if value == "not_applied" else value)
                   for text, value in zip(row["outputs"], row["parsed"])):
                raise ValueError(f"Saved parser output differs at prompt {index}")
            records.append(row)
    return records


def rng_snapshot(torch):
    return dict(cpu=torch.get_rng_state().tolist(),
                cuda=[state.tolist() for state in torch.cuda.get_rng_state_all()])


def restore_rng(torch, state):
    torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
    saved = state["cuda"]
    if len(saved) != torch.cuda.device_count():
        raise ValueError("Visible CUDA device count changed across resume")
    torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in saved])


def generate_official(jobs, config, output):
    """Generate an isolated standalone-checkpoint run; never resumes or overwrites."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation")
    if config["model"] != OFFICIAL_MODEL or config["prompt_count"] != len(jobs):
        raise ValueError("Official generation configuration does not match selected jobs")
    output = create_new_output_directory(output)
    # Preserve the reference run-level RNG behavior: exactly one seed operation,
    # before tokenizer/model loading, and no per-prompt reset.
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        OFFICIAL_CHECKPOINT, local_files_only=True, trust_remote_code=True)
    generation = official_generation_settings(tokenizer)
    loaded = AutoModelForCausalLM.from_pretrained(
        OFFICIAL_CHECKPOINT, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    loaded.eval()
    devices = {str(parameter.device) for parameter in loaded.parameters()}
    if any(not device.startswith("cuda") for device in devices):
        raise ValueError(f"Model is not wholly on CUDA: {sorted(devices)}")
    runtime = dict(
        devices=sorted(devices), dtype=str(loaded.dtype),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_chat_template=tokenizer.chat_template,
        special_tokens_map=tokenizer.special_tokens_map,
        transformers=importlib.metadata.version("transformers"),
        torch=importlib.metadata.version("torch"),
        inherited_generation_config=loaded.generation_config.to_dict(),
        effective_generation=generation)
    (output / "configuration.json").write_text(json.dumps(config, indent=2))
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    _, _, parser, _ = reference_modules()
    segment = output / "segment_000000.jsonl"
    with segment.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for job in jobs:
            tokens, messages = render_official_prompt(tokenizer, job["prompt"])
            input_ids = tokens["input_ids"][0].tolist()
            context = getattr(loaded.config, "max_position_embeddings", None)
            if context is not None and len(input_ids) + GENERATION["max_new_tokens"] > context:
                raise ValueError(f"Prompt {job['prompt_index']} exceeds the model context window")
            rendered = tokenizer.decode(input_ids, skip_special_tokens=False)
            device_tokens = {key: value.to(loaded.device) for key, value in tokens.items()}
            generated = loaded.generate(
                **device_tokens, **generation,
                return_dict_in_generate=True, output_scores=True)
            completion_ids = generated.sequences[:, len(input_ids):]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            completions_special = tokenizer.batch_decode(
                completion_ids, skip_special_tokens=False)
            probabilities = loaded.compute_transition_scores(
                generated.sequences, generated.scores,
                normalize_logits=True).exp().cpu().tolist()
            token_rows = completion_ids.cpu().tolist()
            parsed_values = [parser(text) for text in completions]
            diagnostics = []
            for response_index, (text, token_row, parsed) in enumerate(
                    zip(completions, token_rows, parsed_values)):
                diagnostic = completion_diagnostic(text, token_row)
                diagnostic.update(
                    response_index=response_index,
                    parsed_answer=("NA" if parsed is None else
                                   "Agree" if parsed == 1 else "Disagree"))
                diagnostics.append(diagnostic)
            row = job | dict(
                messages=messages, rendered_chat_input=rendered,
                input_token_ids=input_ids,
                response_indices=list(range(len(completions))),
                generated_completions=completions,
                generated_completions_with_special_tokens=completions_special,
                output_token_ids=token_rows,
                output_token_probabilities=probabilities,
                parsed=["not_applied" if value is None else value for value in parsed_values],
                parsing_diagnostics=diagnostics,
                rng_after=rng_snapshot(torch))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    (output / "COMPLETE").write_text(
        json.dumps({"prompt_blocks": len(jobs),
                    "responses": len(jobs) * GENERATION["num_return_sequences"]}, indent=2))
    print(f"Saved {len(jobs)} official prompt blocks to {output}")


def generate(model, jobs, config, output, stop_after=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generation")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not (output / "configuration.json").exists():
        raise FileExistsError(f"Nonempty output without a matching manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        conf_path = output / "configuration.json"
        if conf_path.exists():
            if json.loads(conf_path.read_text()) != config:
                raise ValueError("Output belongs to a different configuration")
        else:
            with conf_path.open("x") as handle:
                json.dump(config, handle, indent=2)
        _, _, parser, _ = reference_modules()
        records = read_segments(output, jobs, parser)
        if len(records) == len(jobs):
            print(f"Complete: {len(records)} prompt blocks")
            return
        # The authors set the seed once before loading the model. On resume,
        # the exact saved torch RNG state supersedes loading-time RNG use.
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE, local_files_only=True, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto")
        loaded = PeftModel.from_pretrained(base, config["adapter"], is_trainable=False)
        loaded.eval()
        devices = {str(p.device) for p in loaded.parameters()}
        if any(not device.startswith("cuda") for device in devices):
            raise ValueError(f"Model is not wholly on CUDA: {sorted(devices)}")
        runtime = dict(devices=sorted(devices), dtype=str(loaded.dtype),
                       tokenizer_class=type(tokenizer).__name__,
                       tokenizer_chat_template=tokenizer.chat_template,
                       transformers=importlib.metadata.version("transformers"),
                       torch=importlib.metadata.version("torch"),
                       peft=importlib.metadata.version("peft"),
                       inherited_generation_config=loaded.generation_config.to_dict())
        prior = sorted(path for path in output.glob("segment_*.runtime.json")
                       if (output / path.name.replace(".runtime.json", ".jsonl")).exists())
        if prior and json.loads(prior[0].read_text()) != runtime:
            raise ValueError("Runtime changed across resume")
        if records:
            restore_rng(torch, records[-1]["rng_after"])
        occupied = [int(path.name.split("_")[1].split(".")[0])
                    for pattern in ("segment_*.jsonl", "segment_*.runtime.json")
                    for path in output.glob(pattern)]
        segment_number = max(occupied, default=-1) + 1
        segment = output / f"segment_{segment_number:06d}.jsonl"
        with (output / f"segment_{segment_number:06d}.runtime.json").open("x") as handle:
            json.dump(runtime, handle, indent=2)
        written = 0
        with segment.open("x", encoding="utf-8") as handle, torch.inference_mode():
            for job in jobs[len(records):]:
                tokens = tokenizer(job["prompt"], return_tensors="pt")
                ids = tokens["input_ids"][0].tolist()
                context = getattr(loaded.config, "max_position_embeddings", None)
                if context is not None and len(ids) + GENERATION["max_new_tokens"] > context:
                    raise ValueError(f"Prompt {job['prompt_index']} exceeds the model context window")
                tokens = {key: value.to(loaded.device) for key, value in tokens.items()}
                output_ids = loaded.generate(
                    **tokens, **GENERATION, return_dict_in_generate=True, output_scores=True)
                full = tokenizer.batch_decode(output_ids.sequences, skip_special_tokens=True)
                if any(not text.startswith(job["prompt"]) for text in full):
                    raise ValueError(f"Decoded prefix differs from prompt {job['prompt_index']}; "
                                     "the original character-slice parser would be invalid")
                new_ids = output_ids.sequences[:, len(ids):]
                suffix = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                base_model = loaded.get_base_model()
                probabilities = base_model.compute_transition_scores(
                    output_ids.sequences, output_ids.scores, normalize_logits=True).exp().cpu().tolist()
                parsed = [parser(original_answer(text, job["prompt"])) for text in full]
                row = job | dict(input_token_ids=ids, outputs=full, generated_suffixes=suffix,
                                 output_token_ids=new_ids.cpu().tolist(),
                                 output_token_probabilities=probabilities,
                                 parsed=["not_applied" if x is None else x for x in parsed],
                                 rng_after=rng_snapshot(torch))
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
                if stop_after is not None and written >= stop_after:
                    break
        print(f"Saved {written} prompt blocks; {len(records) + written}/{len(jobs)} complete")


def original_policy_functions():
    """Compile unchanged source functions; their module otherwise runs a plotting CLI."""
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    source = SOURCE / "political_biases/domain_specific.py"
    tree = ast.parse(source.read_text())
    util = ast.parse((SOURCE / "political_biases/utils.py").read_text())
    maps = {}
    for node in util.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("id2idx", "policy2idx"):
                maps[node.targets[0].id] = ast.literal_eval(node.value)
    names = {"position_per_statement", "id_to_policy_stance", "positioning_per_model", "dic_category_annotated"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, pd=pd, defaultdict=defaultdict, **maps)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def original_policy_conditions():
    tree = ast.parse((SOURCE / "political_biases/utils.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "test_names" for target in node.targets):
            return list(ast.literal_eval(node.value))
    raise ValueError("Original policy test_names is missing")


def original_policy_frame(scope, matrices, counts, label):
    """Execute the author's numeric plotting statements, excluding print calls."""
    source = SOURCE / "political_biases/domain_specific.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "plot_positioning_per_model")
    model_loop = next(node for node in function.body if isinstance(node, ast.For))
    numeric = model_loop.body[:5]  # agreement through temp_df['std']

    class Quiet(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) \
                    and node.value.func.id == "print":
                return None
            return self.generic_visit(node)

    code = ast.fix_missing_locations(Quiet().visit(ast.Module(body=numeric, type_ignores=[])))
    values = dict(m_model={label: matrices}, model=label, annotated_categories=counts,
                  policy2idx=scope["policy2idx"], pd=scope["pd"], defaultdict=scope["defaultdict"])
    # The policy I/O shim only exposes read_csv; the numeric statements need DataFrame.
    values["pd"] = __import__("pandas")
    exec(compile(code, str(source), "exec"), values)
    return values["temp_df"]


def policy_scores(reliability_dir, model, passed_tests=False):
    """Run original policy functions and extracted numeric plotting statements."""
    import pandas as pd
    scope = original_policy_functions()
    label = model + "-generation"
    files = sorted(Path(reliability_dir).glob(f"{label}_*_statements_reliability_tests.csv"))
    if len(files) != 6:
        raise ValueError("Expected six original template-family reliability files")
    real_read = pd.read_csv
    scope["pd"] = SimpleNamespace(read_csv=lambda path, **kw: real_read(
        ANNOTATIONS if str(path) == "../data/human_annotations/annotations_spiderweb_gold.csv" else path, **kw))
    scope["glob"] = SimpleNamespace(glob=lambda pattern: [str(p) for p in files])
    scope["sorting"] = [label]
    scope["models_renamed"] = {label: label}
    scope["args"] = SimpleNamespace(passed_tests=passed_tests)
    matrices = scope["positioning_per_model"]()[label]
    annotation_counts = scope["dic_category_annotated"]()
    frame = original_policy_frame(scope, matrices, annotation_counts, label)
    rows = []
    condition = "all" if passed_tests is False else "-".join(passed_tests)
    for policy, index in scope["policy2idx"].items():
        for column, (template_id, matrix) in enumerate(matrices.items()):
            agree = matrix["agree"][index]
            disagree = matrix["disagree"][index]
            rows.append(dict(model=model, policy=policy, template_id=template_id,
                             passed_tests=condition, score=float(frame.loc[policy, column]),
                             agree_count=float(agree), disagree_count=float(disagree)))
        rows.append(dict(model=model, policy=policy, template_id="mean",
                         passed_tests=condition,
                         score=float(frame.loc[policy, "mean"]),
                         template_std=float(frame.loc[policy, "std"])))
    return rows


def score(model, jobs, config, input_dir, output_dir):
    import numpy as np
    import pandas as pd
    _, _, parser, binarize = reference_modules()
    from reliability.aggregate import aggregate
    from reliability.evaluation import (compute_all_results,
                                        create_aggregated_questionnaire_cross_template)
    from reliability.utils import add_variante_type_info
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if json.loads((input_dir / "configuration.json").read_text()) != config:
        raise ValueError("Generation configuration changed")
    if len(jobs) != 17208 or config["prompt_count"] != 17208:
        raise ValueError("Original scoring requires the complete benchmark")
    records = read_segments(input_dir, jobs, parser)
    if len(records) != len(jobs):
        raise ValueError(f"Incomplete generation: {len(records)}/{len(jobs)} prompts")
    output_dir.mkdir(parents=True, exist_ok=False)
    compatibility = output_dir / "original_response_format.csv"
    if "gpt" in str(compatibility):
        raise ValueError("The original parser switches templates when its input path contains 'gpt'")
    response_column = model + "-generation_response"
    with compatibility.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["template_id", "statement_id", "prompt", response_column])
        writer.writeheader()
        for row in records:
            for text in row["outputs"]:
                writer.writerow(dict(template_id=row["template_id"], statement_id=row["statement_id"],
                                     prompt=row["prompt"], **{response_column: text}))
    # The reference postprocessor uses paths relative to its own directory.
    previous = Path.cwd()
    try:
        os.chdir(SOURCE / "response_postprocessing")
        parsed = binarize([str(compatibility.resolve())])
    finally:
        os.chdir(previous)
    parsed.to_csv(output_dir / "parsed_responses.csv", index=False)
    expected = [value for row in records for value in row["parsed"]]
    if parsed["binary_answer"].tolist() != expected:
        raise ValueError("Original postprocessor output differs from saved response parsing")
    np.random.seed(42)  # Fixed bootstrap RNG; original scoring formula is unchanged.
    aggregate(parsed, str(output_dir / "aggregated_results.csv"), sample_size=30)
    aggregated = pd.read_csv(output_dir / "aggregated_results.csv", dtype={"statement_id": str})
    annotated = add_variante_type_info(aggregated)
    reliability_dir = output_dir / "reliability"
    reliability_dir.mkdir()
    compute_all_results(annotated, str(reliability_dir))
    cross_template_dir = reliability_dir / "crosstemplate"
    cross_template_dir.mkdir()
    create_aggregated_questionnaire_cross_template(str(reliability_dir), str(cross_template_dir))
    policy_rows = []
    conditions = original_policy_conditions()
    for condition in conditions:
        policy_rows.extend(policy_scores(reliability_dir, model,
                                         condition.split("-") if condition else False))
    pd.DataFrame(policy_rows).to_csv(output_dir / "policy_scores.csv", index=False)
    metadata = dict(generation_config=config, input_dir=str(input_dir.resolve()),
                    original_functions=["format_prompts", "binarise_answer",
                                        "binerarize_answers_two_sided", "aggregate",
                                        "add_variante_type_info", "compute_all_results",
                                        "create_aggregated_questionnaire_cross_template",
                                        "positioning_per_model", "dic_category_annotated"],
                    bootstrap_seed=42, bootstrap_seed_note="Original aggregate does not set a seed; fixed here for repeatability.",
                    response_count=len(parsed), policy_conditions=["all" if not x else x for x in conditions],
                    outputs={str(p.relative_to(output_dir)): digest(p)
                             for p in output_dir.rglob("*") if p.is_file() and p.name != "metadata.json"})
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Scored {len(parsed)} original-format responses for {model}")


def compare_scores(scores_root, score_pattern, output_dir):
    """Tabulate unchanged author scores and descriptive differences from Full."""
    import pandas as pd
    scores_root, output_dir = Path(scores_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Comparison output already exists: {output_dir}")
    source = original_policy_functions()
    policies = set(source["policy2idx"])
    conditions = {"all" if not x else x for x in original_policy_conditions()}
    templates = pd.read_csv(TEMPLATES)
    families = {str(x) for x in templates.loc[templates.model_type == "generation", "template_id"]}
    expected_keys = {(policy, condition, family)
                     for policy in policies for condition in conditions for family in families | {"mean"}}
    tables, manifests = [], {}
    baseline_config = None
    for model in ADAPTERS:
        location = scores_root / score_pattern.format(model=model)
        metadata = json.loads((location / "metadata.json").read_text())
        config = metadata["generation_config"]
        if config["model"] != model or config["prompt_count"] != 17208 \
                or config["reference_commit"] != REFERENCE_COMMIT \
                or metadata["response_count"] != 516240:
            raise ValueError(f"Incomplete or mislabeled scored output: {location}")
        if baseline_config is None:
            baseline_config = config
        for field in ("reference_commit", "source_sha256", "base", "base_config_sha256",
                      "tokenizer_sha256", "prompt_manifest_sha256", "generation", "seed"):
            if config[field] != baseline_config[field]:
                raise ValueError(f"Incomparable {field} for {model}")
        csv_path = location / "policy_scores.csv"
        if metadata["outputs"]["policy_scores.csv"] != digest(csv_path):
            raise ValueError(f"Policy scores changed after scoring: {csv_path}")
        frame = pd.read_csv(csv_path, dtype={"template_id": str})
        columns = ["policy", "passed_tests", "template_id"]
        keys = list(frame[columns].itertuples(index=False, name=None))
        if len(keys) != len(expected_keys) or set(keys) != expected_keys or frame["model"].ne(model).any():
            raise ValueError(f"Incomplete or duplicate policy scores for {model}")
        if frame["score"].isna().any():
            raise ValueError(f"Missing policy scores for {model}")
        tables.append(frame)
        manifests[model] = dict(scored_directory=str(location), metadata_sha256=digest(location / "metadata.json"),
                                policy_scores_sha256=digest(csv_path), adapter=config["adapter"],
                                adapter_sha256=config["adapter_sha256"])
    combined = pd.concat(tables, ignore_index=True)
    means = combined[combined["template_id"] == "mean"].copy()
    baseline = means[means["model"] == "Full"][["policy", "passed_tests", "score"]]
    baseline = baseline.rename(columns={"score": "full_score"})
    differences = means.merge(baseline, on=["policy", "passed_tests"], validate="many_to_one")
    differences["difference_from_full"] = differences["score"] - differences["full_score"]
    output_dir.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output_dir / "combined_policy_scores.csv", index=False)
    differences.to_csv(output_dir / "differences_from_full.csv", index=False)
    with (output_dir / "metadata.json").open("x") as handle:
        json.dump(dict(reference_commit=baseline_config["reference_commit"],
                       method="Descriptive subtraction of unchanged original policy scores; Full baseline",
                       sources=manifests), handle, indent=2)
    print(f"Compared {len(ADAPTERS)} models across {len(policies)} policies and {len(conditions)} conditions")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=["validate", "run", "score", "compare"])
    cli.add_argument("--model", choices=[*ADAPTERS, OFFICIAL_MODEL])
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--input-dir", type=Path)
    cli.add_argument("--scores-root", type=Path)
    cli.add_argument("--score-pattern", default="worldviews_original_{model}_v1_scored")
    cli.add_argument("--limit-prompts", type=int)
    cli.add_argument("--balanced-pilot-statements", type=int,
                     help="OfficialSFT only: whole source statements selected round-robin by country")
    cli.add_argument("--stop-after", type=int)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args()
    if args.action == "compare":
        if not args.scores_root or not args.output_dir:
            cli.error("compare requires --scores-root and --output-dir")
        compare_scores(args.scores_root, args.score_pattern, args.output_dir)
        return
    if args.action == "validate":
        jobs = build_jobs()
        verified = {}
        for name in ADAPTERS:
            path = ROOT / "outputs/sft_runs" / ADAPTERS[name]
            adapter_config = json.loads((path / "adapter_config.json").read_text())
            if Path(adapter_config["base_model_name_or_path"]).resolve() != BASE.resolve():
                raise ValueError(f"{name} uses a different base model")
            if not (path / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(path / "adapter_model.safetensors")
            verified[name] = str(path.resolve())
        print(json.dumps(dict(statements=len({"_".join(j["statement_id"].split("_")[:2]) for j in jobs}),
                              formulations=1434, generation_templates=12,
                              prompt_blocks=len(jobs), generations=len(jobs) * 30,
                              final_template_ids=sorted({j["final_template_id"] for j in jobs}),
                              models=verified), indent=2))
        return
    if not args.model or not args.output_dir:
        cli.error("run/score require --model and --output-dir")
    if args.limit_prompts is not None and args.limit_prompts < 1:
        cli.error("--limit-prompts must be positive")
    if args.balanced_pilot_statements is not None and args.balanced_pilot_statements < 1:
        cli.error("--balanced-pilot-statements must be positive")
    if args.model == OFFICIAL_MODEL:
        if args.action != "run":
            cli.error("OfficialSFT supports the isolated run action; use the distribution summarizer afterward")
        if args.limit_prompts is not None or args.stop_after is not None:
            cli.error("OfficialSFT forbids prefix/stop truncation; use --balanced-pilot-statements")
        jobs = build_jobs("chat")
        selected_source_ids = None
        selected = jobs
        if args.balanced_pilot_statements is not None:
            selected, selected_source_ids = balanced_pilot_jobs(
                jobs, args.balanced_pilot_statements)
        config = official_configuration(selected, selected_source_ids)
        if not args.execute:
            print(json.dumps(dict(configuration=config, first_prompt=selected[0]), indent=2))
            return
        generate_official(selected, config, args.output_dir)
        return
    if args.balanced_pilot_statements is not None:
        cli.error("--balanced-pilot-statements is only valid for OfficialSFT")
    jobs = build_jobs()
    config = configuration(args.model, jobs, args.limit_prompts)
    selected = jobs[:args.limit_prompts] if args.limit_prompts else jobs
    if args.action == "run":
        if not args.execute:
            print(json.dumps(dict(configuration=config, first_prompt=selected[0]), indent=2))
            return
        generate(args.model, selected, config, args.output_dir, args.stop_after)
    else:
        if not args.input_dir:
            cli.error("score requires --input-dir")
        score(args.model, jobs, config, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
