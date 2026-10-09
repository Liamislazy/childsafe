"""Config loading. Paths in the YAML are relative to the repo root.

A config file may start with `_base_: default.yaml` and list only the keys it changes
(e.g. configs/tuned_overrides.yaml written by the grid search).
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    return _merge(_read(path.parent / base), cfg) if base else cfg


def load_config(path: str | Path | None = None) -> dict:
    cfg = _read(Path(path) if path else DEFAULT_CONFIG)
    cfg["paths"] = {k: str((ROOT / v).resolve()) for k, v in cfg["paths"].items()}
    if cfg.get('detector',{}).get('weights'):
        cfg['detector']['weights'] = str((ROOT / cfg['detector']['weights']).resolve())
    return cfg


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
