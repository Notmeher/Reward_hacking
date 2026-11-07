"""Process sandbox for agent commands and grading.

Agent-written code runs as a dedicated unprivileged user (when the parent runs as root, as on
Kaggle) with a scrubbed environment and resource limits, so it cannot read API keys from
the parent's environment, /proc/<pid>/environ, or root-owned output files.
"""
import asyncio
import logging
import os
import pwd
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SANDBOX_USER = "rhsandbox"
DEFAULT_MAX_OUTPUT = 20_000


@dataclass
class ExecResult:
    rc: int
    stdout: str
    stderr: str
    timed_out: bool
    duration: float

    def combined(self, limit: int = DEFAULT_MAX_OUTPUT) -> str:
        out = self.stdout
        if self.stderr.strip():
            out += ("\n" if out and not out.endswith("\n") else "") + self.stderr
        if len(out) > limit:
            out = out[: limit // 2] + f"\n...[{len(out) - limit} chars truncated]...\n" + out[-limit // 2 :]
        if self.timed_out:
            out += "\n[command timed out]"
        return out + f"\n[exit code {self.rc}]"


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def ensure_sandbox_user(name: str = SANDBOX_USER) -> bool:
    """Create the unprivileged sandbox user if running as root. Returns True if usable."""
    if not is_root():
        return False
    try:
        pwd.getpwnam(name)
        return True
    except KeyError:
        pass
    for cmd in (
        ["useradd", "--system", "--no-create-home", "--shell", "/bin/bash", name],
        ["adduser", "--system", "--no-create-home", "--shell", "/bin/bash", name],
    ):
        if shutil.which(cmd[0]) and subprocess.run(cmd, capture_output=True).returncode == 0:
            return True
    log.warning("could not create sandbox user %s", name)
    return False


def _limits(mem_mb: int, fsize_mb: int, nproc: int):
    def apply():
        for res, val in (
            (resource.RLIMIT_AS, mem_mb * 1024 * 1024),
            (resource.RLIMIT_FSIZE, fsize_mb * 1024 * 1024),
            (resource.RLIMIT_CORE, 0),
            (getattr(resource, "RLIMIT_NPROC", None), nproc),
        ):
            if res is None:
                continue
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass

    return apply


async def _read_capped(stream, cap: int) -> str:
    chunks, size = [], 0
    while True:
        data = await stream.read(65536)
        if not data:
            break
        if size < cap:
            chunks.append(data[: cap - size])
        size += len(data)
    text = b"".join(chunks).decode("utf-8", errors="replace")
    if size > cap:
        text += f"\n...[{size - cap} bytes dropped]"
    return text


class LocalSandbox:
    def __init__(
        self,
        base_dir,
        user: str | None = SANDBOX_USER,
        mem_mb: int = 2048,
        fsize_mb: int = 64,
        nproc: int = 512,
        max_concurrency: int = 4,
    ):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.mem_mb, self.fsize_mb, self.nproc = mem_mb, fsize_mb, nproc
        self.max_concurrency = max_concurrency
        self._sems: dict = {}
        self.uid = self.gid = None
        if user and is_root() and ensure_sandbox_user(user):
            pw = pwd.getpwnam(user)
            self.uid, self.gid = pw.pw_uid, pw.pw_gid
        elif user:
            log.warning("sandbox runs as the current user (not root): isolation from secrets is reduced")
        # 0o711: the sandbox user can traverse to its own random-named dirs but not list siblings
        os.chmod(self.base, 0o711)
        self.bin = self.base / "bin"
        self.bin.mkdir(exist_ok=True)
        os.chmod(self.bin, 0o755)
        for name in ("python", "python3"):
            shim = self.bin / name
            if shim.is_symlink() or shim.exists():
                shim.unlink()
            shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
            os.chmod(shim, 0o755)
        self.python = str(self.bin / "python")

    @property
    def isolated(self) -> bool:
        return self.uid is not None

    @property
    def sem(self) -> asyncio.Semaphore:
        # one semaphore per event loop: stages may call asyncio.run() repeatedly
        loop = asyncio.get_running_loop()
        if id(loop) not in self._sems:
            self._sems = {id(loop): asyncio.Semaphore(self.max_concurrency)}
        return self._sems[id(loop)]

    def new_dir(self, prefix: str = "ws") -> Path:
        d = self.base / f"{prefix}-{uuid.uuid4().hex}"
        d.mkdir()
        self.give(d)
        return d

    def give(self, path: Path) -> None:
        """Hand a directory tree to the sandbox user."""
        path = Path(path)
        os.chmod(path, 0o700)
        if self.uid is None:
            return
        for root, dirs, files in os.walk(path):
            os.chown(root, self.uid, self.gid)
            for f in files:
                os.chown(os.path.join(root, f), self.uid, self.gid)

    def env(self, cwd) -> dict:
        return {
            "PATH": f"{self.bin}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(cwd),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "TERM": "dumb",
        }

    async def run(
        self,
        argv,
        cwd,
        timeout: float = 20.0,
        stdin: str | None = None,
        max_output: int = DEFAULT_MAX_OUTPUT * 4,
    ) -> ExecResult:
        if isinstance(argv, str):
            argv = ["/bin/bash", "-c", argv]
        kwargs = {"start_new_session": True, "preexec_fn": _limits(self.mem_mb, self.fsize_mb, self.nproc)}
        if self.uid is not None:
            kwargs.update(user=self.uid, group=self.gid, extra_groups=[])
        async with self.sem:
            t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env=self.env(cwd),
                stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs,
            )
            readers = asyncio.gather(_read_capped(proc.stdout, max_output), _read_capped(proc.stderr, max_output // 4))
            if stdin is not None:
                try:
                    proc.stdin.write(stdin.encode("utf-8", errors="replace"))
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                proc.stdin.close()
            timed_out = False
            try:
                await asyncio.wait_for(proc.wait(), timeout)
            except asyncio.TimeoutError:
                timed_out = True
            # also reaps background processes the command left behind
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            if timed_out:
                try:
                    await asyncio.wait_for(proc.wait(), 5)
                except asyncio.TimeoutError:
                    pass
            try:
                out, err = await asyncio.wait_for(readers, 5)
            except asyncio.TimeoutError:
                out, err = "", "[output collection timed out]"
            rc = proc.returncode if proc.returncode is not None else -9
            return ExecResult(rc, out, err, timed_out, time.monotonic() - t0)

    def remove(self, path) -> None:
        shutil.rmtree(path, ignore_errors=True)
