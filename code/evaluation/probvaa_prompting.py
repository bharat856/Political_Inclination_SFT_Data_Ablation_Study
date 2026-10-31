"""Exact known HF wrapping; missing primary formatter is an explicit dependency."""
import importlib.util
from pathlib import Path
from probvaa_config import MissingReferenceDependency


def get_formatter(mode='model-behavior', dependency=None):
    if mode == 'worldviews-percent':
        # utils/format_prompts.py and prompting/prompts.py: template % statement.
        return lambda template, statement: template % statement
    if not dependency:
        raise MissingReferenceDependency('Primary utils.format_prompt is absent. Supply --formatter /absolute/file.py:function; or explicitly select --format-mode worldviews-percent for the separately documented upstream workflow.')
    filename, function = dependency.rsplit(':', 1)
    module_spec = importlib.util.spec_from_file_location('probvaa_external_formatter', filename)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, function)


def compatible_messages(tokenizer, messages):
    def compatible(test):
        try:
            tokenizer.apply_chat_template(test, return_dict=True, return_tensors='pt', add_generation_prompt=True)
            return True
        except Exception:
            return False
    if not compatible([{'role': 'system', 'content': 'Hello!'}]):
        messages = [{'role': 'user' if m['role'] == 'system' else m['role'], 'content': m['content']} for m in messages]
    if not compatible([{'role': 'user', 'content': 'Hello!'}, {'role': 'user', 'content': 'Hello again!'}]):
        merged = []
        for m in messages:
            if merged and merged[-1]['role'] == m['role']:
                merged[-1]['content'] += '\n' + m['content']
            else:
                merged.append(dict(m))
        messages = [dict(m, content=m['content'].strip()) for m in merged]
    return messages


def tokenize_prompt(tokenizer, prompt, spec):
    if spec.behavior_name == 'tiiuae/falcon-7b-instruct':
        messages = None
        tokens = tokenizer([f'User: {prompt}\nAssistant:'], return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
    elif spec.uses_chat:
        if not tokenizer.chat_template:
            raise MissingReferenceDependency('Instruct/SFT/DPO branch requires an actual tokenizer chat_template. No fallback or invented template is supplied.')
        messages = [{'role': 'user', 'content': prompt}]
        if spec.behavior_name == 'HuggingFaceTB/SmolLM3-3B':
            messages.insert(0, {'role': 'system', 'content': '/no_think'})
        messages = compatible_messages(tokenizer, messages)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors='pt', add_generation_prompt=True)
    else:
        messages = None
        tokens = tokenizer(prompt, return_tensors='pt', return_attention_mask=True, return_token_type_ids=False)
    return tokens, messages


def local_tokenizer_info(spec):
    """Read JSON only; no tokenizer, torch, model, or network initialization."""
    import json
    source = spec.tokenizer_path or spec.base_model
    path = Path(source) / 'tokenizer_config.json'
    info = {'source': source, 'inspected_without_loading': True, 'chat_rendering': 'deferred until manual inference'}
    if path.is_file():
        config = json.loads(path.read_text())
        info.update({k: config.get(k) for k in ('chat_template', 'bos_token', 'eos_token', 'pad_token')})
    return info


def iter_jobs(rows, templates, formatter, samples):
    for t in templates:
        for r in rows:
            prompt = formatter(t['template'], r['statement_text'])
            if not isinstance(prompt, str):
                raise ValueError('The verified OLMo path expects a string prompt')
            for sample in range(samples):
                yield r | {'template_id': t['template_id'], 'template_row': t['template_row'],
                           'template_text': t['template'], 'template_fields': t,
                           'sample_id': sample, 'raw_prompt': prompt}
