#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rel_to(root: Path, value: str) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f'{float(value):.4f}'
    return ''


def parse_unifiedreward_score(output_text: str) -> float | None:
    match = re.search(r"Final Score\s*:\s*([1-5](?:\.[0-9]+)?)", output_text, flags=re.I)
    if match:
        return float(match.group(1))
    numeric = re.fullmatch(r"(?:score\s*[:=]\s*)?([1-5](?:\.[0-9]+)?)\.?", output_text.strip(), flags=re.I)
    return float(numeric.group(1)) if numeric else None


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("ur") if metric == "unifiedreward" else row.get(metric)
    if isinstance(value, int | float):
        return float(value)
    if metric == "unifiedreward" and row.get("ur_raw_output"):
        return parse_unifiedreward_score(str(row["ur_raw_output"]))
    return None


def load_ledgers(run_dir: Path, subdir: str) -> dict[tuple[str, int, int], dict[str, Any]]:
    ledger_path = run_dir / subdir / 'prompt_ledger.jsonl'
    ledger: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in read_jsonl(ledger_path):
        key = (row['condition'], int(row['prompt_index']), int(row['sample_index']))
        ledger[key] = row
    return ledger


def add_rows(
    rows: list[dict[str, Any]],
    run_dir: Path,
    benchmark: str,
    subdir: str,
    metric_rows: list[dict[str, Any]],
    metrics: list[tuple[str, str]],
    old_condition: str,
    new_condition: str,
) -> None:
    ledgers = load_ledgers(run_dir, subdir)
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        grouped[(int(row['prompt_index']), int(row['sample_index']))].setdefault(row['condition'], {}).update(row)

    for key in sorted(grouped):
        pair = grouped[key]
        old_row = pair.get(old_condition)
        new_row = pair.get(new_condition)
        if not old_row or not new_row:
            continue
        prompt_index, sample_index = key
        old_ledger = ledgers.get((old_condition, prompt_index, sample_index), {})
        new_ledger = ledgers.get((new_condition, prompt_index, sample_index), {})
        old_prompt = old_ledger.get('original_prompt') or old_ledger.get('final_prompt') or old_row.get('metric_prompt') or ''
        new_prompt = new_ledger.get('final_prompt') or new_ledger.get('rewritten_prompt') or new_row.get('metric_prompt') or ''
        old_img = rel_to(run_dir, old_row.get('image_path', ''))
        new_img = rel_to(run_dir, new_row.get('image_path', ''))
        for row_key, display_metric in metrics:
            old_score = metric_value(old_row, row_key)
            new_score = metric_value(new_row, row_key)
            if old_score is None or new_score is None:
                continue
            rows.append({
                'benchmark': benchmark,
                'prompt_index': prompt_index,
                'sample_index': sample_index,
                'old_prompt': old_prompt,
                'new_prompt': new_prompt,
                'metric': display_metric,
                'old_score': old_score,
                'new_score': new_score,
                'delta': new_score - old_score,
                'old_img': old_img,
                'new_img': new_img,
            })


def load_all_rows(run_dir: Path, old_condition: str, new_condition: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    geneval_path = run_dir / 'geneval_full' / 'geneval' / 'server_results.jsonl'
    if geneval_path.exists():
        add_rows(
            rows,
            run_dir,
            'GenEval full',
            'geneval_full',
            read_jsonl(geneval_path),
            [('score', 'geneval_score'), ('reward', 'geneval_reward'), ('strict_reward', 'geneval_strict_reward')],
            old_condition,
            new_condition,
        )
    ocr_path = run_dir / 'ocr1k' / 'metrics_preference.jsonl'
    if ocr_path.exists():
        add_rows(
            rows,
            run_dir,
            'OCR1k',
            'ocr1k',
            read_jsonl(ocr_path),
            [('ocr', 'ocr')],
            old_condition,
            new_condition,
        )
    pref_path = run_dir / 'pickscore_sfw' / 'metrics_preference.jsonl'
    if pref_path.exists():
        add_rows(
            rows,
            run_dir,
            'PickScore-SFW',
            'pickscore_sfw',
            read_jsonl(pref_path),
            [('pickscore', 'pickscore'), ('hps', 'hps'), ('unifiedreward', 'unifiedreward')],
            old_condition,
            new_condition,
        )
    return rows


def load_rewrite_meta(run_dir: Path) -> tuple[str, str]:
    config_path = run_dir / 'geneval_full' / 'run_config_rewrite.json'
    if not config_path.exists():
        return 'unknown', 'unknown'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    specs = config.get('rewriter_specs', {})
    model = specs.get('qwen25_vl_3b', {}).get('model_id', 'unknown')
    settings = (
        'PromptRL user-only enhancement template; no system prompt; '
        f"temperature={config.get('rewrite_temperature')}; "
        f"top_p={config.get('rewrite_top_p')}; "
        f"max_new_tokens={config.get('rewrite_max_new_tokens')}"
    )
    return model, settings


def load_generator_meta(run_dir: Path) -> str:
    manifest = run_dir / 'run_manifest.json'
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        gen = data.get('generator', {})
        model = gen.get('model_path', 'unknown')
        backend = gen.get('backend')
        steps = gen.get('steps')
        guidance = gen.get('guidance')
        size = f"{gen.get('width')}x{gen.get('height')}"
        parts = [str(model)]
        if backend:
            parts.append(f'backend={backend}')
        parts.append(f'size={size}')
        parts.append(f'steps={steps}')
        parts.append(f'guidance={guidance}')
        return '; '.join(parts)
    return 'unknown'


def build_html(run_dir: Path, rows: list[dict[str, Any]], old_condition: str, new_condition: str) -> str:
    rewrite_model, rewrite_settings = load_rewrite_meta(run_dir)
    generator = load_generator_meta(run_dir)
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    benchmarks = sorted({row['benchmark'] for row in rows})
    metrics = sorted({row['metric'] for row in rows})
    for row in rows:
        deltas[(row['benchmark'], row['metric'])].append(float(row['delta']))
    pills = []
    for (benchmark, metric), vals in sorted(deltas.items()):
        mean_delta = sum(vals) / len(vals)
        pills.append(f'<span class="pill"><b>{html.escape(benchmark)}</b> {html.escape(metric)} mean delta {mean_delta:+.4f}</span>')

    body_rows = []
    for row in rows:
        delta = float(row['delta'])
        cls = 'pos' if delta > 0 else 'neg' if delta < 0 else 'zero'
        old_img = html.escape(row['old_img'])
        new_img = html.escape(row['new_img'])
        body_rows.append(
            '<tr '
            f'data-benchmark="{html.escape(row["benchmark"])}" '
            f'data-metric="{html.escape(row["metric"])}" '
            f'data-delta="{delta}">'
            f'<td><div class="bench">{html.escape(row["benchmark"])}</div><div class="idx">idx {row["prompt_index"]}</div></td>'
            f'<td class="prompt">{html.escape(str(row["old_prompt"]))}</td>'
            f'<td class="prompt">{html.escape(str(row["new_prompt"]))}</td>'
            f'<td>{html.escape(row["metric"])}</td>'
            f'<td class="num">{fmt(row["old_score"])}</td>'
            f'<td class="num">{fmt(row["new_score"])}</td>'
            f'<td class="num {cls}">{fmt(delta)}</td>'
            f'<td class="imgcell"><a href="{old_img}" target="_blank"><img loading="lazy" src="{old_img}" alt="old image"></a></td>'
            f'<td class="imgcell"><a href="{new_img}" target="_blank"><img loading="lazy" src="{new_img}" alt="new image"></a></td>'
            '</tr>'
        )

    benchmark_options = ''.join(f'<option>{html.escape(v)}</option>' for v in benchmarks)
    metric_options = ''.join(f'<option>{html.escape(v)}</option>' for v in metrics)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PromptRL Rewrite Benchmark Comparison</title><style>
:root {{ color-scheme: light; --bg:#f7f7f4; --panel:#ffffff; --ink:#1c1d1f; --muted:#666b73; --line:#d8dadf; --head:#eceff3; --pos:#086b38; --neg:#9f1d1d; }}
* {{ box-sizing: border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }}
header {{ position:sticky; top:0; z-index:5; background:rgba(247,247,244,.96); border-bottom:1px solid var(--line); padding:16px 20px 12px; }}
h1 {{ margin:0 0 8px; font-size:22px; line-height:1.2; letter-spacing:0; }} .meta {{ display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap:8px 16px; color:var(--muted); margin-bottom:12px; }} .meta b {{ color:var(--ink); }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }} input, select, button {{ height:34px; border:1px solid var(--line); border-radius:6px; background:white; color:var(--ink); padding:0 10px; }} input {{ min-width:360px; flex:1; }} button {{ cursor:pointer; }}
.summary {{ display:flex; flex-wrap:wrap; gap:6px; padding:12px 20px 8px; }} .pill {{ display:inline-flex; gap:6px; align-items:center; background:#fff; border:1px solid var(--line); border-radius:999px; padding:5px 9px; color:var(--muted); }}
main {{ padding:0 20px 24px; }} table {{ width:100%; border-collapse:separate; border-spacing:0; background:var(--panel); border:1px solid var(--line); table-layout:fixed; }} th, td {{ border-bottom:1px solid var(--line); border-right:1px solid var(--line); vertical-align:top; padding:8px; }} th {{ position:sticky; top:136px; background:var(--head); z-index:3; text-align:left; font-size:12px; text-transform:uppercase; color:#3b4048; }} th:last-child, td:last-child {{ border-right:0; }} tr:last-child td {{ border-bottom:0; }}
.col-bench {{ width:150px; }} .col-prompt {{ width:24%; }} .col-metric {{ width:116px; }} .col-score {{ width:82px; }} .col-img {{ width:150px; }} .prompt {{ max-height:168px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; }} .bench {{ font-weight:700; }} .idx {{ color:var(--muted); font-size:12px; margin-top:3px; }} .num {{ text-align:right; font-variant-numeric: tabular-nums; }} .pos {{ color:var(--pos); font-weight:700; }} .neg {{ color:var(--neg); font-weight:700; }} .imgcell img {{ display:block; width:132px; height:132px; object-fit:cover; border:1px solid var(--line); border-radius:6px; background:#eee; }} .hidden {{ display:none; }}
@media (max-width: 900px) {{ .meta {{ grid-template-columns:1fr; }} input {{ min-width:180px; }} th {{ top:216px; }} table {{ min-width:1180px; }} main {{ overflow:auto; }} }}
</style></head><body><header><h1>PromptRL Rewrite Benchmark Comparison</h1><div class="meta"><div><b>Run</b>: {html.escape(run_dir.name)}</div><div><b>Prompt mapping</b>: old = <code>{html.escape(old_condition)}</code>, new = <code>{html.escape(new_condition)}</code></div><div><b>Rewrite LM</b>: {html.escape(rewrite_model)}</div><div><b>Rewrite settings</b>: {html.escape(rewrite_settings)}</div><div><b>Generator</b>: {html.escape(generator)}</div><div><b>Rows</b>: {len(rows)} prompt/metric comparisons</div><div><b>Images</b>: click thumbnails to open full PNG</div></div><div class="controls"><input id="q" type="search" placeholder="Filter prompts, benchmark, metric, or index..."><select id="benchmark"><option value="">All benchmarks</option>{benchmark_options}</select><select id="metric"><option value="">All metrics</option>{metric_options}</select><button id="positive">Positive deltas</button><button id="clear">Clear</button><span id="count"></span></div></header><section class="summary">{''.join(pills)}</section><main><table id="tbl"><thead><tr><th class="col-bench">Benchmark</th><th class="col-prompt">Old prompt</th><th class="col-prompt">New prompt</th><th class="col-metric">Metric</th><th class="col-score">Old score</th><th class="col-score">New score</th><th class="col-score">Delta</th><th class="col-img">Old img</th><th class="col-img">New img</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table></main><script>
const q=document.getElementById('q'),benchmark=document.getElementById('benchmark'),metric=document.getElementById('metric'),count=document.getElementById('count');let positiveOnly=false;function applyFilter(){{const needle=q.value.trim().toLowerCase(),b=benchmark.value,m=metric.value;let shown=0;for(const tr of document.querySelectorAll('#tbl tbody tr')){{const show=(!needle||tr.innerText.toLowerCase().includes(needle))&&(!b||tr.dataset.benchmark===b)&&(!m||tr.dataset.metric===m)&&(!positiveOnly||Number(tr.dataset.delta)>0);tr.classList.toggle('hidden',!show);if(show)shown++;}}count.textContent=`${{shown}} shown`;}}q.addEventListener('input',applyFilter);benchmark.addEventListener('change',applyFilter);metric.addEventListener('change',applyFilter);document.getElementById('positive').addEventListener('click',()=>{{positiveOnly=!positiveOnly;applyFilter();}});document.getElementById('clear').addEventListener('click',()=>{{q.value='';benchmark.value='';metric.value='';positiveOnly=false;applyFilter();}});applyFilter();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description='Build row-level PromptRL comparison HTML from completed image calibration outputs.')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--old-condition', default='none')
    parser.add_argument('--new-condition', default='qwen25_vl_3b')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_all_rows(run_dir, args.old_condition, args.new_condition)
    if not rows:
        raise SystemExit(f'No comparison rows found under {run_dir}')
    output = Path(args.output) if args.output else run_dir / 'comparison_table.html'
    output.write_text(build_html(run_dir, rows, args.old_condition, args.new_condition), encoding='utf-8')
    print(f'[comparison] wrote {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()
