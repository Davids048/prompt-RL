#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import pickle
import socket
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def run(
    records: list[dict[str, Any]],
    trial: dict[str, Any],
    run_dir: Path,
) -> list[dict[str, Any]]:
    import requests
    from PIL import Image

    task = trial["task"]
    params = task.get("eval_params", {}).get("geneval", {})
    server_url = os.environ.get("GENEVAL_URL") or params.get("server_url", "http://127.0.0.1:18085")
    batch_size = int(params.get("batch_size", 32))
    timeout = float(os.environ.get("GENEVAL_TIMEOUT") or params.get("timeout", 900))
    startup_timeout = float(os.environ.get("GENEVAL_STARTUP_TIMEOUT") or params.get("startup_timeout", timeout))
    pad_multiple = int(params.get("pad_multiple", 0))
    eval_dir = run_dir / "eval" / trial["trial_id"] / "geneval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    raw_path = eval_dir / "server_results.jsonl"

    successes = [row for row in records if row.get("artifact_path") and not row.get("error")]
    if not successes:
        return records

    results_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    server_process: subprocess.Popen[Any] | None = None
    try:
        server_process = ensure_server(server_url, startup_timeout)
        for chunk_start in range(0, len(successes), batch_size):
            chunk = successes[chunk_start:chunk_start + batch_size]
            image_bytes: list[bytes] = []
            metadatas: list[dict[str, Any]] = []
            for row in chunk:
                image = Image.open(run_dir / row["artifact_path"]).convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                image_bytes.append(buffer.getvalue())
                metadatas.append(row["prompt_metadata"])
            real_count = len(chunk)
            if pad_multiple and real_count % pad_multiple:
                pad_count = pad_multiple - (real_count % pad_multiple)
                image_bytes.extend([image_bytes[-1]] * pad_count)
                metadatas.extend([metadatas[-1]] * pad_count)
            payload = {
                "images": image_bytes,
                "meta_datas": metadatas,
                "only_strict": False,
            }
            response = requests.post(server_url, data=pickle.dumps(payload), timeout=timeout)
            response.raise_for_status()
            data = pickle.loads(response.content)
            for idx, row in enumerate(chunk):
                result = {
                    "trial_id": row["trial_id"],
                    "prompt_index": row["prompt_index"],
                    "sample_index": row["sample_index"],
                    "artifact_path": row["artifact_path"],
                    "score": float(data["scores"][idx]),
                    "reward": float(data["rewards"][idx]),
                    "strict_reward": float(data["strict_rewards"][idx]),
                    "source": str(raw_path.relative_to(run_dir)),
                }
                results_by_key[(int(row["prompt_index"]), int(row["sample_index"]))] = result
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        message = repr(exc)
        out: list[dict[str, Any]] = []
        for row in records:
            item = dict(row)
            if item.get("artifact_path") and not item.get("error"):
                item["eval"] = {}
                item["error"] = {
                    "stage": "eval",
                    "eval": "geneval",
                    "message": message,
                }
            out.append(item)
        return out
    finally:
        stop_server(server_process)

    out = []
    for row in records:
        item = dict(row)
        key = (int(item["prompt_index"]), int(item["sample_index"]))
        result = results_by_key.get(key)
        if result:
            item.setdefault("eval", {})
            item["eval"]["geneval"] = {
                "score": result["score"],
                "reward": result["reward"],
                "strict_reward": result["strict_reward"],
                "source": result["source"],
            }
        out.append(item)
    return out


def ensure_server(server_url: str, timeout: float) -> subprocess.Popen[Any] | None:
    if server_reachable(server_url):
        return None

    gunicorn = os.environ.get("GENEVAL_GUNICORN")
    if not gunicorn:
        return None

    cwd = Path(os.environ.get("GENEVAL_SERVER_CWD", "."))
    log_path = os.environ.get("GENEVAL_SERVER_LOG")
    log_file = None
    stdout = subprocess.DEVNULL
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "ab")  # noqa: SIM115
        stdout = log_file

    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    process = None
    try:
        process = subprocess.Popen(  # noqa: S603
            [gunicorn, "--bind", f"{host}:{port}", "app_geneval:create_app()"],
            cwd=str(cwd),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        if log_file is not None:
            log_file.close()
        wait_for_server(server_url, timeout)
    except Exception:
        if log_file is not None and not log_file.closed:
            log_file.close()
        if process is not None:
            stop_server(process)
        raise
    return process


def stop_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def wait_for_server(server_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_reachable(server_url):
            return
        time.sleep(5)
    raise RuntimeError(f"GenEval reward server did not become reachable before timeout: {server_url}")


def server_reachable(server_url: str) -> bool:
    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False
