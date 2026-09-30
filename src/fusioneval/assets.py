"""Local defaults only. Downloading is always an explicit user action."""
from pathlib import Path
from .errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {'qwen3-4b', 'qwen3-8b', 'llama3.2-3b'}
DOMAINS = ('math', 'code', 'agent', 'if', 'science')


def model_path(profile, role):
    if profile is None or profile.name not in MODEL_NAMES:
        raise ConfigError('Set an explicit model path, or select the supplied model_profile for default paths')
    return str(ROOT / 'models' / profile.name / role)


def data_path(domain):
    return str(ROOT / 'data' / 'calibration' / 'data' / (domain + '.jsonl'))


def missing_hint(path):
    """Explain a missing default input; explicit custom paths are never replaced."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        return 'Set the correct local path in your experiment config.'
    parts = path.relative_to(ROOT).parts
    if len(parts) >= 2 and parts[0] == 'models' and parts[1] in MODEL_NAMES:
        if len(parts) >= 3 and parts[2] == 'initial':
            return 'Run TA first and set initial_model to its output, or save that result at this path.'
        return 'Download first: python scripts/download/models.py --model ' + parts[1]
    if len(parts) >= 2 and parts[0] == 'data' and parts[1] in {'calibration', 'sft', 'mopd', 'student'}:
        return 'Download first: python scripts/download/data.py --dataset ' + parts[1]
    return 'Prepare this input locally or correct its path in your experiment config.'
