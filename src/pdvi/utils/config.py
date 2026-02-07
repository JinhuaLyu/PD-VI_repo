from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root_from(file_path: str | Path, levels: int = 3) -> Path:
    p = Path(file_path).resolve()
    for _ in range(levels):
        p = p.parent
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping at top-level: {path}")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
