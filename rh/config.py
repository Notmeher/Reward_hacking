import copy
import os
from pathlib import Path

import yaml


def load_config(path, overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        parent = load_config(Path(path).parent / base)
        cfg = deep_merge(parent, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value) -> None:
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur.setdefault(p, {})
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def load_env_file(path) -> None:
    """Minimal .env loader; existing environment variables win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise KeyError(f"Environment variable {name} is not set (add it as a Kaggle Secret or to .env)")
    return val
