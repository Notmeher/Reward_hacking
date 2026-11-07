"""Local SWE-bench helpers: check the swebench/docker setup and pre-pull instance images."""
import asyncio
import logging
from pathlib import Path

from rh.stages.common import filter_tasks, load_or_build_tasks
from rh.swe import docker, harness
from rh.swe.env import image_name

log = logging.getLogger(__name__)


def _tasks(cfg, out_root):
    return [t for t in filter_tasks(load_or_build_tasks(cfg, out_root), cfg.get("task_filter")) if t.kind == "swebench"]


def preflight(cfg: dict, out_root: Path) -> bool:
    tasks = _tasks(cfg, out_root)
    if not tasks:
        log.error("no swebench tasks in config")
        return False
    t = tasks[0]
    spec = harness.make_spec(t.meta["instance"])
    script = harness.eval_script(spec)
    log.info("swebench OK: eval script for %s has %d lines", t.task_id, len(script.splitlines()))
    proxy = harness.proxy_script(script, t.meta["instance"]["base_commit"])
    log.info("proxy script drops %d lines (test reset + test patch)", len(script.splitlines()) - len(proxy.splitlines()))
    swe = cfg.get("env", {}).get("swe", {})
    tmpl = swe.get("image_template", "swebench/sweb.eval.x86_64.{iid_1776}:latest")

    async def check():
        r = await docker._run(["docker", "info", "--format", "{{.Architecture}} {{.MemTotal}}"], timeout=60)
        if r.rc != 0:
            log.error("docker not reachable: %s", r.stderr[-300:])
            return False
        log.info("docker: %s", r.stdout.strip())
        present = [await docker.image_exists(image_name(x.task_id, tmpl)) for x in tasks]
        log.info("%d/%d instance images present locally (run `rh swe-pull` to fetch the rest)", sum(present), len(tasks))
        return True

    return asyncio.run(check())


def pull(cfg: dict, out_root: Path, concurrency: int = 2) -> None:
    swe = cfg.get("env", {}).get("swe", {})
    tmpl = swe.get("image_template", "swebench/sweb.eval.x86_64.{iid_1776}:latest")
    platform = swe.get("platform", "linux/amd64")
    tasks = _tasks(cfg, out_root)

    async def run():
        sem = asyncio.Semaphore(concurrency)

        async def one(t):
            img = image_name(t.task_id, tmpl)
            if await docker.image_exists(img):
                return
            async with sem:
                r = await docker.pull(img, platform)
                log.info("%s %s", "pulled" if r.rc == 0 else "FAILED", img)

        await asyncio.gather(*(one(t) for t in tasks))

    asyncio.run(run())
