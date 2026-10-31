"""Inference implementation. Imported only after explicit --execute and asset gates.

Nothing in this module loads models at import time. No calls made during setup.
"""
import importlib.metadata
from probvaa_config import EFFECTIVE_GENERATION, MissingReferenceDependency
from probvaa_prompting import tokenize_prompt


def package_versions():
    result = {}
    for name in ('transformers', 'torch', 'peft', 'accelerate', 'numpy', 'pandas'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class Generator:
    def __init__(self, spec, seed=42, allow_download=False, dtype=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        self.seed, self.spec, self.set_seed = seed, spec, set_seed
        set_seed(seed)
        kwargs = {'trust_remote_code': True, 'local_files_only': not allow_download}
        self.tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_path or spec.base_model, **kwargs)
        if spec.uses_chat and not self.tokenizer.chat_template:
            raise MissingReferenceDependency('Selected tokenizer lacks chat_template. Supply the verified instructed tokenizer explicitly; no fallback is invented.')
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        # Transformers 5 forwards unknown loading kwargs to the model constructor.
        # return_dict_in_generate belongs to generate(), not from_pretrained().
        load_kwargs = dict(kwargs, device_map='auto', output_hidden_states=False)
        if dtype:
            import torch
            load_kwargs['torch_dtype'] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(spec.base_model, **load_kwargs)
        if spec.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
        self.model.eval()
        self.settings = EFFECTIVE_GENERATION | {'pad_token_id': self.pad_token_id, 'eos_token_id': self.tokenizer.eos_token_id,
                                                'return_dict_in_generate': False}
        self.load_settings = load_kwargs

    def metadata(self):
        return {'tokenizer_source': self.spec.tokenizer_path or self.spec.base_model,
                'chat_template': self.tokenizer.chat_template,
                'special_tokens_map': self.tokenizer.special_tokens_map,
                'tokenizer_class': type(self.tokenizer).__name__,
                'generation_kwargs': self.settings,
                'inherited_model_generation_config': self.model.generation_config.to_dict(),
                'load_settings': {k: str(v) for k, v in self.load_settings.items()},
                'model_dtype': str(self.model.dtype), 'package_versions': package_versions()}

    def generate(self, prompt):
        import torch
        self.set_seed(self.seed)  # The reference resets the SAME seed for every call.
        tokens, messages = tokenize_prompt(self.tokenizer, prompt, self.spec)
        input_ids = tokens['input_ids'][0].tolist()
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        with torch.inference_mode():
            generated = self.model.generate(**tokens, **self.settings)
        output_ids = generated[0][len(input_ids):].tolist()
        return {'raw_output': self.tokenizer.decode(output_ids, skip_special_tokens=True),
                'raw_output_with_special_tokens': self.tokenizer.decode(output_ids, skip_special_tokens=False),
                'rendered_model_input': self.tokenizer.decode(input_ids, skip_special_tokens=False),
                'input_token_ids': input_ids, 'output_token_ids': output_ids,
                'messages': messages, 'generation_parameters': self.settings}
