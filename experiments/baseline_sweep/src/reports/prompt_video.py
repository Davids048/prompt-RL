#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


PREFERRED_METRICS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def write_html_report(run_dir: Path, output_path: Path | None = None) -> Path:
    output = output_path or run_dir / "prompt_video_report.html"
    records = latest_records(read_records(run_dir / "records.jsonl"))
    rows = [row for row in records if row.get("prompt_index") is not None and row.get("artifact_path")]
    if not rows:
        raise ValueError(f"No artifact records found under {run_dir}")

    trial_order = load_trial_order(run_dir / "run_plan.jsonl")
    metrics = metric_columns(rows)
    groups = group_by_prompt(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    data_dir = output.parent / f"{output.stem}_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / f"{output.stem}_manifest.json"

    manifest = write_group_data(run_dir, output, data_dir, groups, metrics, trial_order)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.write_text(build_html(os.path.relpath(manifest_path, output.parent)), encoding="utf-8")
    return output


def read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("trial_id"), row.get("prompt_index"), row.get("sample_index"))
        if None in key:
            passthrough.append(row)
            continue
        latest[key] = row
    return passthrough + list(latest.values())


def load_trial_order(path: Path) -> dict[str, int]:
    order: dict[str, int] = {}
    if not path.exists():
        return order
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            trial_id = row.get("trial_id")
            if trial_id:
                order[str(trial_id)] = idx
    return order


def metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for row in rows:
        for eval_result in row.get("eval", {}).values():
            if not isinstance(eval_result, dict):
                continue
            for key, value in eval_result.items():
                if key == "source":
                    continue
                if isinstance(value, int | float):
                    seen.add(str(key))
    preferred = [metric for metric in PREFERRED_METRICS if metric in seen]
    extras = sorted(seen.difference(preferred))
    return preferred + extras


def group_by_prompt(rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_index = int(row["prompt_index"])
        original_prompt = str(row.get("original_prompt") or "")
        groups[(prompt_index, original_prompt)].append(row)
    return groups


def write_group_data(
    run_dir: Path,
    output_path: Path,
    data_dir: Path,
    groups: dict[tuple[int, str], list[dict[str, Any]]],
    metrics: list[str],
    trial_order: dict[str, int],
) -> dict[str, Any]:
    prompts: list[dict[str, Any]] = []
    total_rows = 0
    for (prompt_index, original_prompt), rows in sorted(groups.items(), key=lambda item: item[0][0]):
        sorted_rows = sorted(rows, key=lambda row: row_sort_key(row, trial_order))
        group_path = data_dir / f"prompt_{prompt_index:06d}.json"
        group_payload = {
            "prompt_index": prompt_index,
            "original_prompt": original_prompt,
            "rows": [report_row(run_dir, output_path, row) for row in sorted_rows],
        }
        group_path.write_text(json.dumps(group_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        rel_group_path = os.path.relpath(group_path, output_path.parent)
        total_rows += len(sorted_rows)
        prompts.append({
            "prompt_index": prompt_index,
            "original_prompt": original_prompt,
            "row_count": len(sorted_rows),
            "data_path": rel_group_path,
        })
    return {
        "run": str(run_dir),
        "prompt_count": len(prompts),
        "row_count": total_rows,
        "metrics": metrics,
        "prompts": prompts,
    }


def row_sort_key(row: dict[str, Any], trial_order: dict[str, int]) -> tuple[int, str, int]:
    trial_id = str(row.get("trial_id") or "")
    enhancer = str(row.get("enhancer_alias") or "")
    sample_index = int(row.get("sample_index") or 0)
    return (trial_order.get(trial_id, 10_000), enhancer, sample_index)


def report_row(run_dir: Path, output_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    generation_prompt = str(row.get("generation_prompt") or row.get("enhanced_prompt") or row.get("original_prompt") or "")
    return {
        "prompt_id": prompt_id(row),
        "original_prompt": str(row.get("original_prompt") or ""),
        "enhancer": str(row.get("enhancer_alias") or "none"),
        "sample_index": int(row.get("sample_index") or 0),
        "generation_prompt": generation_prompt,
        "scores": numeric_scores(row),
        "video_path": artifact_href(run_dir, output_path, str(row.get("artifact_path") or "")),
    }


def numeric_scores(row: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for eval_result in row.get("eval", {}).values():
        if not isinstance(eval_result, dict):
            continue
        for key, value in eval_result.items():
            if key == "source":
                continue
            if isinstance(value, int | float):
                scores[str(key)] = float(value)
    return scores


def prompt_id(row: dict[str, Any]) -> str:
    prompt_index = int(row.get("prompt_index") or 0)
    source_index = row.get("prompt_source_index")
    if source_index is None:
        return f"#{prompt_index:04d}"
    return f"#{prompt_index:04d}\nsource {source_index}"


def artifact_href(run_dir: Path, output_path: Path, value: str) -> str:
    path = Path(value)
    artifact = path if path.is_absolute() else run_dir / path
    return os.path.relpath(artifact, output_path.parent)


def build_html(manifest_href: str) -> str:
    return HTML_TEMPLATE.replace("__MANIFEST_URL__", json.dumps(manifest_href))


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prompt Video Report</title>
<style>
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --panel: #ffffff;
  --ink: #1d2939;
  --muted: #667085;
  --line: #d0d5dd;
  --head: #eef2f7;
  --divider: #1f3a5f;
  --button: #175cd3;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  position: sticky;
  top: 0;
  z-index: 5;
  padding: 16px 22px 14px;
  border-bottom: 1px solid var(--line);
  background: #fbfcfe;
}
h1 {
  margin: 0 0 8px;
  font-size: 24px;
  line-height: 1.2;
  letter-spacing: 0;
}
.meta, .controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  align-items: center;
}
.meta { color: var(--muted); margin-bottom: 12px; }
.meta b { color: var(--ink); }
.controls { gap: 8px; }
button, input {
  height: 36px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fff;
  color: var(--ink);
  font: inherit;
}
button {
  min-width: 92px;
  padding: 0 12px;
  cursor: pointer;
}
button.primary {
  border-color: var(--button);
  background: var(--button);
  color: #fff;
}
button:disabled {
  cursor: default;
  opacity: 0.45;
}
label {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  color: var(--muted);
}
input {
  width: 120px;
  padding: 0 10px;
}
main { padding: 22px; }
.prompt-group {
  border-top: 8px solid var(--divider);
  background: var(--panel);
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
}
.prompt-heading {
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-top: 0;
  background: #ffffff;
}
.prompt-heading h2 {
  margin: 0 0 5px;
  font-size: 18px;
  line-height: 1.25;
  letter-spacing: 0;
}
.prompt-heading .prompt-text {
  color: var(--muted);
  overflow-wrap: anywhere;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-top: 0;
}
table {
  min-width: 1880px;
  width: max-content;
  border-collapse: collapse;
  table-layout: fixed;
}
th, td {
  padding: 8px;
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  vertical-align: top;
}
th {
  background: var(--head);
  color: #344054;
  font-size: 12px;
  font-weight: 700;
  text-align: left;
  text-transform: uppercase;
}
td:last-child, th:last-child { border-right: 0; }
tr:last-child td { border-bottom: 0; }
.prompt-id { width: 110px; font-variant-numeric: tabular-nums; white-space: pre-line; }
.original-prompt { width: 330px; }
.enhancer { width: 230px; overflow-wrap: anywhere; }
.sample { width: 84px; text-align: right; font-variant-numeric: tabular-nums; }
.generation-prompt { width: 560px; }
.score { width: 122px; text-align: right; font-variant-numeric: tabular-nums; }
.video { width: 360px; }
.enhancer-tone-0 { background: #ffffff; }
.enhancer-tone-1 { background: #f3f4f6; }
.best-enhancer { background: #dcfce7; }
.enhancer-tone-0 .enhancer,
.enhancer-tone-1 .enhancer,
.best-enhancer .enhancer { font-weight: 700; }
.metric-best {
  color: #000000;
  font-weight: 800;
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-underline-offset: 3px;
}
.prompt-cell {
  max-height: 190px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.muted { color: var(--muted); }
.score-missing { color: var(--muted); text-align: center; }
video {
  display: block;
  width: 340px;
  max-width: 100%;
  aspect-ratio: 16 / 9;
  background: #111827;
  border: 1px solid var(--line);
}
.error {
  padding: 16px;
  color: #9f1d1d;
  background: #fff;
  border: 1px solid #f4b3b3;
}
</style>
</head>
<body>
<header>
  <h1>Prompt Video Report</h1>
  <div class="meta">
    <span><b>Run</b>: <span id="runName"></span></span>
    <span><b>Original prompts</b>: <span id="promptCount"></span></span>
    <span><b>Total rows</b>: <span id="rowCount"></span></span>
    <span><b>Current rows</b>: <span id="currentRows"></span></span>
    <span><b>Green rows</b>: best enhancer by equal-weight normalized measured metrics</span>
  </div>
  <div class="controls">
    <button id="prevBtn" class="primary" type="button">Prev</button>
    <button id="nextBtn" class="primary" type="button">Next</button>
    <label>Prompt index <input id="promptIndex" type="number" min="0" step="1"></label>
    <span id="position" class="muted"></span>
  </div>
</header>
<main>
  <section id="viewer" class="prompt-group">
    <div class="prompt-heading">
      <h2 id="promptTitle"></h2>
      <div id="promptText" class="prompt-text"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr id="headerRow"></tr></thead>
        <tbody id="bodyRows"></tbody>
      </table>
    </div>
  </section>
</main>
<script>
const MANIFEST_URL = __MANIFEST_URL__;
let manifest = null;
let currentPosition = 0;

const els = {
  runName: document.getElementById("runName"),
  promptCount: document.getElementById("promptCount"),
  rowCount: document.getElementById("rowCount"),
  currentRows: document.getElementById("currentRows"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  promptIndex: document.getElementById("promptIndex"),
  position: document.getElementById("position"),
  promptTitle: document.getElementById("promptTitle"),
  promptText: document.getElementById("promptText"),
  headerRow: document.getElementById("headerRow"),
  bodyRows: document.getElementById("bodyRows"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function metricLabel(metric) {
  return metric.replaceAll("_", " ");
}

function makeHeader() {
  const fixed = ["Prompt ID", "Original prompt", "Enhancer", "Sample", "Generation prompt"];
  const metricHeads = manifest.metrics.map(metricLabel);
  const all = fixed.concat(metricHeads).concat(["Video"]);
  els.headerRow.innerHTML = all.map((name) => `<th>${escapeHtml(name)}</th>`).join("");
}

function isNumeric(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function bestEnhancersForPrompt(rows) {
  const stats = new Map();
  for (const row of rows) {
    const stat = stats.get(row.enhancer) ?? new Map();
    for (const metric of manifest.metrics) {
      const value = row.scores[metric];
      if (isNumeric(value)) {
        const metricStat = stat.get(metric) ?? {sum: 0, count: 0};
        metricStat.sum += value;
        metricStat.count += 1;
        stat.set(metric, metricStat);
      }
    }
    stats.set(row.enhancer, stat);
  }

  const bestMeanByMetric = new Map();
  for (const metric of manifest.metrics) {
    let bestMean = -Infinity;
    for (const stat of stats.values()) {
      const metricStat = stat.get(metric);
      if (metricStat && metricStat.count > 0) {
        bestMean = Math.max(bestMean, metricStat.sum / metricStat.count);
      }
    }
    if (bestMean > -Infinity) {
      bestMeanByMetric.set(metric, bestMean);
    }
  }

  const normalizedStats = new Map();
  for (const [enhancer, stat] of stats.entries()) {
    let sum = 0;
    let count = 0;
    for (const [metric, metricStat] of stat.entries()) {
      const bestMean = bestMeanByMetric.get(metric);
      if (bestMean === undefined || metricStat.count === 0) {
        continue;
      }
      const mean = metricStat.sum / metricStat.count;
      sum += bestMean === 0 ? (mean === 0 ? 1 : 0) : mean / bestMean;
      count += 1;
    }
    normalizedStats.set(enhancer, {sum, count});
  }

  let bestMean = -Infinity;
  for (const stat of normalizedStats.values()) {
    if (stat.count > 0) {
      bestMean = Math.max(bestMean, stat.sum / stat.count);
    }
  }
  return new Set(
    Array.from(normalizedStats.entries())
      .filter(([, stat]) => stat.count > 0 && Math.abs((stat.sum / stat.count) - bestMean) < 1e-12)
      .map(([enhancer]) => enhancer)
  );
}

function bestMetricScores(rows) {
  const best = new Map();
  for (const metric of manifest.metrics) {
    let maxValue = -Infinity;
    for (const row of rows) {
      const value = row.scores[metric];
      if (isNumeric(value)) {
        maxValue = Math.max(maxValue, value);
      }
    }
    if (maxValue > -Infinity) {
      best.set(metric, maxValue);
    }
  }
  return best;
}

async function loadPrompt(position) {
  if (!manifest) return;
  currentPosition = Math.max(0, Math.min(position, manifest.prompts.length - 1));
  const meta = manifest.prompts[currentPosition];
  const response = await fetch(meta.data_path);
  if (!response.ok) {
    throw new Error(`Could not load ${meta.data_path}: HTTP ${response.status}`);
  }
  const group = await response.json();
  renderPrompt(meta, group);
}

function renderPrompt(meta, group) {
  els.promptTitle.innerHTML = `Original Prompt ${currentPosition + 1} <span class="muted">(index ${meta.prompt_index})</span>`;
  els.promptText.textContent = meta.original_prompt;
  els.currentRows.textContent = String(meta.row_count);
  els.promptIndex.value = String(meta.prompt_index);
  els.position.textContent = `${currentPosition + 1} of ${manifest.prompts.length}`;
  els.prevBtn.disabled = currentPosition === 0;
  els.nextBtn.disabled = currentPosition === manifest.prompts.length - 1;

  const enhancerOrder = [];
  for (const row of group.rows) {
    if (!enhancerOrder.includes(row.enhancer)) {
      enhancerOrder.push(row.enhancer);
    }
  }
  const enhancerTone = new Map(enhancerOrder.map((enhancer, index) => [enhancer, index % 2]));
  const bestEnhancers = bestEnhancersForPrompt(group.rows);
  const metricBest = bestMetricScores(group.rows);

  els.bodyRows.innerHTML = group.rows.map((row) => {
    const tone = enhancerTone.get(row.enhancer) ?? 0;
    const rowClass = bestEnhancers.has(row.enhancer) ? "best-enhancer" : `enhancer-tone-${tone}`;
    const scores = manifest.metrics.map((metric) => {
      const value = row.scores[metric];
      if (value === undefined || value === null) {
        return '<td class="score-missing">not evaluated</td>';
      }
      const bestValue = metricBest.get(metric);
      const bestClass = bestValue !== undefined && Math.abs(value - bestValue) < 1e-12 ? " metric-best" : "";
      return `<td class="score${bestClass}">${Number(value).toFixed(6)}</td>`;
    }).join("");
    return `<tr class="${rowClass}">
      <td class="prompt-id">${escapeHtml(row.prompt_id)}</td>
      <td class="original-prompt"><div class="prompt-cell">${escapeHtml(row.original_prompt)}</div></td>
      <td class="enhancer">${escapeHtml(row.enhancer)}</td>
      <td class="sample">${escapeHtml(row.sample_index)}</td>
      <td class="generation-prompt"><div class="prompt-cell">${escapeHtml(row.generation_prompt)}</div></td>
      ${scores}
      <td class="video"><video controls autoplay loop muted preload="auto" playsinline src="${escapeHtml(row.video_path)}"></video></td>
    </tr>`;
  }).join("");
  history.replaceState(null, "", `#prompt=${meta.prompt_index}`);
}

function positionForPromptIndex(promptIndex) {
  return manifest.prompts.findIndex((prompt) => prompt.prompt_index === promptIndex);
}

function goToPromptIndex() {
  const wanted = Number(els.promptIndex.value);
  const position = positionForPromptIndex(wanted);
  if (position >= 0) {
    loadPrompt(position).catch(showError);
  }
}

function initialPosition() {
  const match = location.hash.match(/prompt=(\\d+)/);
  if (!match) return 0;
  const position = positionForPromptIndex(Number(match[1]));
  return position >= 0 ? position : 0;
}

function showError(error) {
  els.bodyRows.innerHTML = `<tr><td class="error" colspan="${manifest ? manifest.metrics.length + 6 : 6}">${escapeHtml(error.message)}</td></tr>`;
}

async function init() {
  const response = await fetch(MANIFEST_URL);
  if (!response.ok) {
    throw new Error(`Could not load ${MANIFEST_URL}: HTTP ${response.status}`);
  }
  manifest = await response.json();
  els.runName.textContent = manifest.run;
  els.promptCount.textContent = String(manifest.prompt_count);
  els.rowCount.textContent = String(manifest.row_count);
  els.promptIndex.max = String(Math.max(0, manifest.prompt_count - 1));
  makeHeader();
  await loadPrompt(initialPosition());
}

els.prevBtn.addEventListener("click", () => loadPrompt(currentPosition - 1).catch(showError));
els.nextBtn.addEventListener("click", () => loadPrompt(currentPosition + 1).catch(showError));
els.promptIndex.addEventListener("change", goToPromptIndex);
els.promptIndex.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    goToPromptIndex();
  }
});

init().catch(showError);
</script>
</body>
</html>
"""
