import hashlib
import json
import os
import threading
from pathlib import Path

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(os.path.abspath(path), threading.Lock())


def append_jsonl(path, record: dict) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # a session killed mid-write leaves a truncated last line
                continue
    return out


def read_jsonl_glob(directory, pattern: str = "*.jsonl") -> list[dict]:
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.rglob(pattern)):
        out.extend(read_jsonl(p))
    return out


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"  # per-process so parallel pipeline steps never share a temp file
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def stable_hash(obj, n: int = 12) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
