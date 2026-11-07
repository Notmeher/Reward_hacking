"""Command-line entry point: `rh <stage> --config CONFIG [--out OUT] [-o key=value ...]`."""
import argparse
import logging
import os
import sys
from pathlib import Path

from rh.config import load_config, load_env_file


def _out(args) -> Path:
    return Path(args.out or os.environ.get("RH_OUT") or "rh_out").resolve()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rh")
    p.add_argument("stage", choices=["tasks", "sweep", "monitor", "report", "grpo", "generalize", "preflight",
                                     "pipeline", "setup-sandbox", "swe-preflight", "swe-pull"])
    p.add_argument("--config", "-c")
    p.add_argument("--out", help="output root (default $RH_OUT or ./rh_out)")
    p.add_argument("--max-hours", type=float, default=float(os.environ.get("RH_MAX_HOURS", 11.0)))
    p.add_argument("--limit", type=int, default=None, help="cap cells per model (smoke tests)")
    p.add_argument("-o", "--override", action="append", default=[], help="dotted override, e.g. grid.seeds=1")
    p.add_argument("--env-file", default=".env", help="local .env with API keys (ignored if missing)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    for noisy in ("httpx", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env_file(args.env_file)
    if os.environ.get("HF_API_KEY") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]

    if args.stage == "setup-sandbox":
        from rh.env.sandbox import ensure_sandbox_user, is_root

        ok = ensure_sandbox_user()
        print("sandbox user ready" if ok else f"sandbox user unavailable (root={is_root()})")
        return 0 if ok or not is_root() else 1
    if not args.config:
        p.error("--config is required")
    cfg = load_config(args.config, args.override)
    out = _out(args)
    out.mkdir(parents=True, exist_ok=True)

    if args.stage == "tasks":
        from rh.stages.common import tasks_path
        from rh.stages.tasks import build_and_save

        build_and_save(cfg["tasks"], tasks_path(out, cfg), cfg)
    elif args.stage == "sweep":
        from rh.stages import sweep

        sweep.main(cfg, out, args.max_hours, args.limit)
    elif args.stage == "monitor":
        from rh.stages import monitor

        monitor.main(cfg, out, args.max_hours)
    elif args.stage == "report":
        from rh.stages import report

        report.main(cfg, out)
    elif args.stage == "grpo":
        from rh.stages import grpo

        grpo.main(cfg, out, args.max_hours)
    elif args.stage == "generalize":
        from rh.stages import generalize

        generalize.main(cfg, out, args.max_hours)
    elif args.stage == "preflight":
        from rh.stages import preflight

        return 0 if preflight.main(cfg) else 1
    elif args.stage == "pipeline":
        from rh.stages import pipeline

        results = pipeline.main(cfg, out, args.max_hours)
        return 0 if all(r["rc"] == 0 for r in results.values()) else 1
    elif args.stage == "swe-preflight":
        from rh.stages import swe_tools

        return 0 if swe_tools.preflight(cfg, out) else 1
    elif args.stage == "swe-pull":
        from rh.stages import swe_tools

        swe_tools.pull(cfg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
