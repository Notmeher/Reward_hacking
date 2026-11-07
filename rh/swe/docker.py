"""Thin async wrapper around the docker CLI for per-episode containers."""
import asyncio
import shlex
import time
import uuid

from rh.env.sandbox import ExecResult, _read_capped


async def _run(argv, stdin: str | None = None, timeout: float = 600, cap: int = 200_000) -> ExecResult:
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    readers = asyncio.gather(_read_capped(proc.stdout, cap), _read_capped(proc.stderr, cap // 4))
    if stdin is not None:
        proc.stdin.write(stdin.encode("utf-8", errors="replace"))
        await proc.stdin.drain()
        proc.stdin.close()
    timed_out = False
    try:
        await asyncio.wait_for(proc.wait(), timeout)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        await proc.wait()
    out, err = await readers
    return ExecResult(proc.returncode if proc.returncode is not None else -9, out, err, timed_out, time.monotonic() - t0)


class Container:
    def __init__(self, image: str, platform: str | None = None, memory: str = "4g", network: str = "none",
                 workdir: str = "/testbed"):
        self.image, self.platform, self.memory, self.network, self.workdir = image, platform, memory, network, workdir
        self.name = f"rh-{uuid.uuid4().hex[:12]}"
        self.started = False

    async def start(self, timeout: float = 1800) -> None:
        argv = ["docker", "run", "-d", "--name", self.name, "--network", self.network, "--memory", self.memory,
                "-w", self.workdir]
        if self.platform:
            argv += ["--platform", self.platform]
        argv += [self.image, "sleep", "infinity"]
        r = await _run(argv, timeout=timeout)
        if r.rc != 0:
            raise RuntimeError(f"docker run failed for {self.image}: {r.stderr[-800:]}")
        self.started = True

    async def exec(self, cmd: str, timeout: float = 60, stdin: str | None = None, cap: int = 200_000) -> ExecResult:
        inner = f"timeout -k 5 {int(timeout)} bash -lc {shlex.quote(cmd)}"
        argv = ["docker", "exec"] + (["-i"] if stdin is not None else []) + ["-w", self.workdir, self.name, "bash", "-c", inner]
        r = await _run(argv, stdin=stdin, timeout=timeout + 30, cap=cap)
        if r.rc == 124:
            r.timed_out = True
        return r

    async def write_file(self, path: str, content: str) -> ExecResult:
        cmd = f'mkdir -p "$(dirname {shlex.quote(path)})" && cat > {shlex.quote(path)}'
        return await self.exec(cmd, timeout=60, stdin=content)

    async def read_file(self, path: str, cap: int = 5_000_000) -> str:
        r = await self.exec(f"cat {shlex.quote(path)}", timeout=120, cap=cap)
        return r.stdout

    async def stop(self) -> None:
        if self.started:
            await _run(["docker", "rm", "-f", self.name], timeout=120)
            self.started = False


async def image_exists(image: str) -> bool:
    r = await _run(["docker", "image", "inspect", image], timeout=60)
    return r.rc == 0


async def pull(image: str, platform: str | None = None) -> ExecResult:
    argv = ["docker", "pull"] + (["--platform", platform] if platform else []) + [image]
    return await _run(argv, timeout=7200)
