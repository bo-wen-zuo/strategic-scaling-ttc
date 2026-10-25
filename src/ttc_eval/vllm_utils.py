"""
Launches vLLM's own OpenAI-compatible server (`vllm serve`) rather than a
custom FastAPI wrapper, so that `/v1/chat/completions` applies the model's
chat template for us. Generation scripts send {"role": "user", ...} messages
and let vLLM render them -- they never hand-build a raw text prompt string
for a chat model.
"""

import time
import requests
import subprocess
import socket
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ttc_eval.utils import GenerateScriptConfig


def gpus_to_cvd(gpus: list[int]) -> str:
    return "CUDA_VISIBLE_DEVICES=" + ",".join(str(g) for g in gpus)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_health(port: int, popen: subprocess.Popen, retry_seconds=2, max_retries=500):
    print(f"Waiting for vLLM server to start on port {port}...")
    for i in range(max_retries):
        try:
            resp = requests.get(f"http://localhost:{port}/health")
            if resp.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass

        if popen.poll() is not None:
            raise RuntimeError(f"vLLM server died with code {popen.returncode} before starting.")

        print(f"vLLM server not yet ready (attempt {i}), retrying...")
        time.sleep(retry_seconds)

    raise RuntimeError(f"vLLM server not ready after {max_retries} attempts.")


@contextmanager
def vllm_manager(config: "GenerateScriptConfig"):
    gpus = config.gpus
    if isinstance(gpus, int):
        gpus = [gpus]
    elif isinstance(gpus, str):
        gpus = [int(g) for g in gpus.split(",")]

    extra_args = " ".join(str(a) for a in (config.vllm_args or []))

    port = find_free_port()

    vllm_command = f"vllm serve {config.model} --port {port} {extra_args}"
    if gpus is not None:
        vllm_command = f"{gpus_to_cvd(gpus)} {vllm_command} --tensor-parallel-size {len(gpus)}"

    print(f"Starting vLLM server: {vllm_command}")
    vllm_process = subprocess.Popen(vllm_command, shell=True)
    print(f"Started vLLM server with pid {vllm_process.pid}")

    try:
        wait_for_health(port, vllm_process)
        yield port
    finally:
        print(f"Killing vLLM server (pid {vllm_process.pid})...")
        subprocess.run(f"pkill -P -9 {vllm_process.pid}", shell=True)
        try:
            os.kill(vllm_process.pid, 9)
        except ProcessLookupError:
            pass
        print("Done killing vLLM server.")
