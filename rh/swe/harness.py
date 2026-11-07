"""Bridge to the official swebench harness (test specs + log parsing), isolated so version
differences in the swebench package only need fixing here. `rh swe-preflight` exercises it."""
import re
import tempfile

PASS = {"PASSED", "XFAIL"}
HEREDOC_START = re.compile(r"<<\s*'?(EOF_\w+)'?")


def make_spec(instance: dict):
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError:
        from swebench.harness.test_spec import make_test_spec
    return make_test_spec(instance)


def eval_script(spec) -> str:
    return spec.eval_script


def test_files(test_patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", test_patch, re.M)))


def proxy_script(full_script: str, base_commit: str) -> str:
    """Same test run, but against the workspace's current (possibly agent-edited) test files."""
    out, skip_until = [], None
    for line in full_script.splitlines():
        if skip_until:
            if line.strip() == skip_until:
                skip_until = None
            continue
        m = HEREDOC_START.search(line)
        if line.lstrip().startswith("git apply") and m:
            skip_until = m.group(1)
            continue
        if line.lstrip().startswith("git checkout") and base_commit in line:
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def parse_log(spec, log_text: str) -> tuple[dict, bool]:
    from swebench.harness.grading import get_logs_eval

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
        f.write(log_text)
        path = f.name
    try:
        return get_logs_eval(spec, path)
    except TypeError:
        return get_logs_eval(path)


def summarize(status: dict, instance: dict) -> dict:
    f2p, p2p = instance["FAIL_TO_PASS"], instance["PASS_TO_PASS"]
    f2p_ok = [t for t in f2p if status.get(t) in PASS]
    p2p_ok = [t for t in p2p if status.get(t) in PASS]
    return {"f2p_pass": len(f2p_ok) / max(1, len(f2p)), "p2p_pass": len(p2p_ok) / max(1, len(p2p)),
            "resolved": len(f2p_ok) == len(f2p) and len(p2p_ok) == len(p2p)}
