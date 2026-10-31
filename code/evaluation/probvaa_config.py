"""Reference provenance and explicit model/asset configuration (no ML imports)."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / 'references/political_content_training_data'
WORLDVIEWS = ROOT / 'references/eval_political_worldviews'
EXPECTED_COMMITS = {
    str(PRIMARY): '335a0bcc96acab3ce601c957d5cb53bc0f87566d',
    str(WORLDVIEWS): 'c06dd493330bebe37ef9d06f77e90dc3e13e0960',
}
DIMENSIONS = (
    'open_foreign_policy', 'liberal_economic_policy',
    'restrictive_financial_policy', 'law_and_order',
    'restrictive_migration_policy', 'expanded_environ_protection',
    'expanded_social_welfare_state', 'liberal_society',
)
DECLARED_GENERATION = dict(do_sample=False, max_new_tokens=10, temperature=1.0, top_p=None)
# _HFModel.__call__ overrides do_sample when temperature is nonzero.
EFFECTIVE_GENERATION = dict(DECLARED_GENERATION, do_sample=True, num_return_sequences=1)


class MissingReferenceDependency(ValueError):
    pass


@dataclass
class ModelSpec:
    model: str
    model_type: str
    base_model: str
    adapter_path: str | None = None
    tokenizer_path: str | None = None
    prompt_model_name: str | None = None

    @property
    def behavior_name(self):
        return self.prompt_model_name or self.base_model

    @property
    def uses_chat(self):
        n = self.behavior_name
        return n == 'HuggingFaceTB/SmolLM3-3B' or any(k in n.lower() for k in ('instruct', 'dpo', 'sft'))

    def metadata(self):
        return asdict(self) | {'behavior_name': self.behavior_name, 'uses_chat': self.uses_chat}


def model_spec(preset='full', base_model=None, adapter=None, tokenizer=None, prompt_model_name=None):
    presets = {
        'full': (str(ROOT / 'hf_cache/allenai_Olmo-3-1025-7B'), str(ROOT / 'outputs/sft_runs/olmo3_instruct_full_4gpu_v4')),
        'official': ('allenai/Olmo-3-7B-Instruct-SFT', None),
        'base': (str(ROOT / 'hf_cache/allenai_Olmo-3-1025-7B'), None),
    }
    base, default_adapter = presets[preset]
    base = base_model or base
    adapter = str((ROOT / adapter).resolve()) if adapter else default_adapter
    return ModelSpec(preset, 'peft_adapter' if adapter else 'hf_model', base,
                     adapter, tokenizer, prompt_model_name)


def references():
    result = []
    for path, expected in EXPECTED_COMMITS.items():
        commit = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
        remote = subprocess.check_output(['git', '-C', path, 'remote', 'get-url', 'origin'], text=True).strip()
        if commit != expected:
            raise ValueError(f'Reference changed: {path}: expected {expected}, found {commit}')
        result.append({'path': path, 'commit': commit, 'origin': remote,
                       'upstream_url': 'https://github.com/tceron/' + Path(path).name + '.git'})
    return result


def fingerprint(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
