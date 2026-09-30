"""Readable automatic run names; explicit output_root/output remain supported."""
from pathlib import Path
import re
from uuid import uuid4

from .assets import ROOT
from .errors import ConfigError

ALIASES = {'task_arithmetic': 'ta', 'average': 'soup', 'recursive_average': 'soup'}
TAGS = {
    'ta': [('scale', 's')],
    'ties': [('scale', 's'), ('density', 'd')],
    'dare': [('scale', 's'), ('drop_rate', 'drop')],
    'della': [('scale', 's'), ('drop_rate', 'drop'), ('window', 'w')],
    'localize_stitch': [('density', 'd')],
    'fisher': [('examples', 'n'), ('samples', 'samples')],
    'adamerging': [('initial_coefficient', 'init'), ('learning_rate', 'lr'), ('steps', 'steps')],
    'regmean': [('alpha', 'a'), ('examples', 'n')],
    'featcal': [('ridge_lambda', 'lam'), ('anchor_rho', 'rho'), ('teacher_alpha', 'a')],
    'surgery': [('rank', 'r'), ('learning_rate', 'lr'), ('steps', 'steps')],
}


def output_path(root_value, name, directory, model, method, count, parameters, continual=False):
    if name:
        if not isinstance(name, str):
            raise ConfigError('output must be a path string, or null for automatic naming')
        target = Path(name).expanduser()
        return (target if target.is_absolute() else directory / target).resolve()
    method = ALIASES.get(method, method)
    root = Path(root_value).expanduser() if root_value else ROOT / 'outputs'
    if not root.is_absolute():
        root = directory / root
    model = re.sub(r'[^a-z0-9.-]+', '-', model.lower()).strip('-.')[:32] or 'model'
    tags = [model, 'm' + str(count)]
    for key, label in TAGS.get(method, []):
        if key in parameters:
            value = parameters[key]
            tags.append(label + (format(value, '.6g') if isinstance(value, (int, float)) else str(value)))
    name = '_'.join(tags) + '_' + uuid4().hex[:8]
    return (root / ('continual' if continual else '') / method / name).resolve()
