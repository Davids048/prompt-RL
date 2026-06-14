"""Validate FastVideo request-level data parallelism."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def _get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _service_urls(config: dict[str, Any]) -> list[str]:
    host = str(_get(config, "fastvideo_service.public_host"))
    count = int(_get(config, "fastvideo_service.data_parallel_workers", 1))
    raw_ports = _get(config, "fastvideo_service.worker_ports")
    if raw_ports is None:
        base_port = int(_get(config, "fastvideo_service.port"))
        ports = [base_port + index for index in range(count)]
    else:
        ports = [int(port) for port in raw_ports]
    return [f"http://{host}:{port}" for port in ports]


def _generator_config(config: dict[str, Any], *, probe_steps: int) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "model": generation["model"],
        "height": int(generation["height"]),
        "width": int(generation["width"]),
        "fps": int(generation["fps"]),
        "num_frames": int(generation["num_frames"]),
        "num_inference_steps": probe_steps,
        "guidance_scale": float(generation["guidance_scale"]),
        "flow_shift": float(generation["flow_shift"]),
        "negative_prompt": generation["negative_prompt"],
    }


def _route_index(request_id: str, worker_count: int) -> int:
    digest = hashlib.blake2s(request_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % worker_count


def _request_id_for_worker(prefix: str, worker_index: int, worker_count: int) -> str:
    # Pick ids that exercise the same stable route function as the Slime hook.
    for counter in range(10000):
        request_id = f"{prefix}_w{worker_index}_{counter:04d}"
        if _route_index(request_id, worker_count) == worker_index:
            return request_id
    raise RuntimeError(f"could not find request id for worker {worker_index}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {error.code}: {body}") from error


def _wait_for_health(urls: list[str], *, timeout: float, retries: int, interval: float) -> list[dict[str, Any]]:
    last_error = ""
    for _attempt in range(1, retries + 1):
        rows = []
        failed = False
        for url in urls:
            health_url = f"{url}/health"
            try:
                rows.append({"url": health_url, "response": _get_json(health_url, timeout=timeout)})
            except Exception as error:  # noqa: BLE001
                failed = True
                last_error = f"{health_url}: {error}"
                break
        if not failed:
            return rows
        time.sleep(interval)
    raise RuntimeError(f"FastVideo workers did not become healthy: {last_error}")


def _probe_requests(config: dict[str, Any], urls: list[str], *, probe_steps: int) -> list[dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"fastvideo_dp_probe_{timestamp}"
    generator = _generator_config(config, probe_steps=probe_steps)
    requests = []
    for worker_index, url in enumerate(urls):
        request_id = _request_id_for_worker(prefix, worker_index, len(urls))
        requests.append(
            {
                "url": f"{url}/generate_and_score",
                "route_index": worker_index,
                "payload": {
                    "request_id": request_id,
                    "original_prompt": "a single red cube on a clean white table",
                    "enhanced_prompt": "a realistic studio photograph of a single red cube on a clean white table",
                    "artifact_kind": "image",
                    "comparison_group_id": f"{prefix}_group_{worker_index}",
                    "seed": 930000 + worker_index,
                    "generator": generator,
                },
            }
        )
    return requests


def _run_probe(requests: list[dict[str, Any]], *, timeout: float) -> list[dict[str, Any]]:
    def send(row: dict[str, Any]) -> dict[str, Any]:
        response = _post_json(row["url"], row["payload"], timeout=timeout)
        return {
            "url": row["url"],
            "route_index": row["route_index"],
            "request_id": row["payload"]["request_id"],
            "response": response,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(send, row) for row in requests]
        return [future.result() for future in futures]


def _summarize(
    output_dir: Path,
    urls: list[str],
    health_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> bool:
    health_workers = [row["response"].get("worker", {}) for row in health_rows]
    response_workers = [row["response"].get("worker", {}) for row in response_rows]
    health_worker_ids = {str(worker.get("worker_id")) for worker in health_workers}
    response_worker_ids = {str(worker.get("worker_id")) for worker in response_workers}
    health_devices = {str(worker.get("cuda_visible_devices")) for worker in health_workers}
    response_devices = {str(worker.get("cuda_visible_devices")) for worker in response_workers}
    statuses = [row["response"].get("status") for row in response_rows]
    passed = (
        len(urls) > 1
        and len(health_worker_ids) == len(urls)
        and len(response_worker_ids) == len(urls)
        and all(status == "completed" for status in statuses)
    )

    lines = [
        "# FastVideo Data-Parallel Validation",
        "",
        f"- Worker URLs: `{', '.join(urls)}`",
        f"- Health worker ids: `{sorted(health_worker_ids)}`",
        f"- Health CUDA devices: `{sorted(health_devices)}`",
        f"- Probe worker ids: `{sorted(response_worker_ids)}`",
        f"- Probe CUDA devices: `{sorted(response_devices)}`",
        f"- Probe statuses: `{statuses}`",
        f"- Result: `{'PASS' if passed else 'FAIL'}`",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--health-timeout", type=float, default=5.0)
    parser.add_argument("--health-retries", type=int, default=60)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--probe-timeout", type=float, default=1800.0)
    parser.add_argument("--probe-steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_yaml(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = _service_urls(config)

    health_rows = _wait_for_health(
        urls,
        timeout=args.health_timeout,
        retries=args.health_retries,
        interval=args.health_interval,
    )
    _write_jsonl(output_dir / "health.jsonl", health_rows)

    requests = _probe_requests(config, urls, probe_steps=args.probe_steps)
    _write_jsonl(output_dir / "probe_requests.jsonl", requests)
    responses = _run_probe(requests, timeout=args.probe_timeout)
    _write_jsonl(output_dir / "probe_responses.jsonl", responses)

    return 0 if _summarize(output_dir, urls, health_rows, responses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
