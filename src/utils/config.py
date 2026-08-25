# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
# Documentation reviewed and improved for clarity.
"""Config loader — reads pipeline_config.yaml and exposes a typed namespace."""

from __future__ import annotations
import yaml
from pathlib import Path
from types import SimpleNamespace


def _to_namespace(d):
    """Recursively convert dict → SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(i) for i in d]
    return d


def load_config(path: str | Path = None) -> SimpleNamespace:
    """
    Load YAML config. Defaults to configs/pipeline_config.yaml
    relative to the repo root.
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "pipeline_config.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)
