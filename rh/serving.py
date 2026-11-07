"""Launch/stop vLLM OpenAI-compatible servers (one per GPU = data parallel).

vLLM runs from its own venv (config `python`) so its pinned torch never clashes with the
training stack. Servers require an API key (passed via env, not argv) so sandboxed agent code
on localhost cannot call or reconfigure them.
"""
import logging
import os
import secrets
import signal
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


class VLLMServer:
    def __init__(self, cfg: dict, gpu, port: int, log_path: Path, api_key: str):
        # gpu: int (one GPU) or list of ints (tensor parallel across them)
        self.gpus = [int(g) for g in gpu] if isinstance(gpu, (list, tuple)) else [int(gpu)]
        self.cfg, self.gpu, self.port, self.log_path, self.api_key = cfg, gpu, port, log_path, api_key
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def argv(self) -> list[str]:
        c = self.cfg
        a = [
            c.get("python", "python"), "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(c["model"]),
            "--served-model-name", c.get("served_name", str(c["model"])),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--dtype", c.get("dtype", "half"),
            "--max-model-len", str(c.get("max_model_len", 16384)),
            "--gpu-memory-utilization", str(c.get("gpu_memory_utilization", 0.9)),
            "--max-num-seqs", str(c.get("max_num_seqs", 32)),
        ]
        if len(self.gpus) > 1:
            a += ["--tensor-parallel-size", str(len(self.gpus))]
        if c.get("quantization"):
            a += ["--quantization", c["quantization"]]
        if c.get("tool_call_parser"):
            a += ["--enable-auto-tool-choice", "--tool-call-parser", c["tool_call_parser"]]
        if c.get("reasoning_parser"):
            a += ["--reasoning-parser", c["reasoning_parser"]]
        if c.get("enable_lora"):
            a += ["--enable-lora", "--max-lora-rank", str(c.get("max_lora_rank", 32)),
                  "--max-loras", str(c.get("max_loras", 2))]
            for name, path in (c.get("lora_modules") or {}).items():
                a += ["--lora-modules", f"{name}={path}"]
        if c.get("enable_prefix_caching", True):
            a += ["--enable-prefix-caching"]
        a += [str(x) for x in c.get("extra_args", [])]
        return a

    def start(self) -> None:
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in self.gpus), "VLLM_API_KEY": self.api_key})
        if self.cfg.get("enable_lora"):
            env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        for k, v in (self.cfg.get("env") or {}).items():
            env[k] = str(v)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log_path, "ab")
        log.info("starting vLLM on GPU %s port %s: %s", self.gpu, self.port, " ".join(self.argv()))
        self.proc = subprocess.Popen(self.argv(), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self, timeout: float) -> None:
        t0 = time.monotonic()
        headers = {"Authorization": f"Bearer {self.api_key}"}
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM on GPU {self.gpu} exited with {self.proc.returncode}:\n{self.tail()}")
            try:
                if httpx.get(f"{self.base_url}/models", headers=headers, timeout=5).status_code == 200:
                    log.info("vLLM on GPU %s ready after %.0fs", self.gpu, time.monotonic() - t0)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM on GPU {self.gpu} not ready after {timeout}s:\n{self.tail()}")

    def tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(self.proc.pid, sig)
                    self.proc.wait(wait)
                    break
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    continue

    # runtime LoRA swapping (requires enable_lora)
    def load_lora(self, name: str, path: str) -> None:
        h = {"Authorization": f"Bearer {self.api_key}"}
        r = httpx.post(f"http://127.0.0.1:{self.port}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path},
                       headers=h, timeout=300)
        if r.status_code != 200:
            raise RuntimeError(f"load_lora_adapter failed: {r.status_code} {r.text[:300]}")

    def unload_lora(self, name: str) -> None:
        h = {"Authorization": f"Bearer {self.api_key}"}
        httpx.post(f"http://127.0.0.1:{self.port}/v1/unload_lora_adapter", json={"lora_name": name}, headers=h, timeout=120)


class ServerGroup:
    def __init__(self, cfg: dict, log_dir: Path):
        self.api_key = secrets.token_hex(16)
        port0 = int(cfg.get("port_base", 8100))
        self.servers = [
            VLLMServer(cfg, gpu, port0 + i, Path(log_dir) / f"vllm_gpu{'-'.join(map(str, gpu)) if isinstance(gpu, (list, tuple)) else gpu}.log",
                       self.api_key)
            for i, gpu in enumerate(cfg.get("gpus", [0]))
        ]
        self.timeout = float(cfg.get("startup_timeout", 1500))

    def __enter__(self):
        try:
            for s in self.servers:
                s.start()
            for s in self.servers:
                s.wait_ready(self.timeout)
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        for s in self.servers:
            s.stop()

    @property
    def base_urls(self) -> list[str]:
        return [s.base_url for s in self.servers]
