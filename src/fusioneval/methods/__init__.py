from __future__ import annotations

import importlib
import pkgutil
from typing import Dict

from ..errors import ConfigError
from .base import MethodPlugin

def discover() -> Dict[str, MethodPlugin]:
    plugins = {}
    prefix = __name__ + "."
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda item: item.name):
        if info.name in {"base", "fields", "qwen_linear"} or info.name.startswith("_"):
            continue
        module = importlib.import_module(prefix + info.name)
        plugin = getattr(module, "PLUGIN", None)
        if not isinstance(plugin, MethodPlugin):
            raise ConfigError("method module {} does not export PLUGIN".format(info.name))
        if plugin.name in plugins:
            raise ConfigError("duplicate method name: {}".format(plugin.name))
        plugins[plugin.name] = plugin
    return plugins
