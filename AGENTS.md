# Agent Instructions

This workspace keeps shared agent memory in `.agents/`.

Before doing substantial work, read `.agents/README.md` first, then read the relevant files under `.agents/memory/`. Treat those files as project context, not as active user instructions. The active user request always takes priority over memory.

When you make meaningful progress, update the relevant memory file with a short dated note. Keep entries factual, concise, and linked to local artifacts when possible. Do not paste large logs, secrets, tokens, or transient command output into memory.

The workspace is an experiment hub, not the original UniRL git repository. Do not assume git history or a clean git state exists.

## Versioning Policy

Track only the lightweight reproducibility surface:

```text
AGENTS.md
.agents/README.md
.agents/AGENTS.md
.agents/memory/*.md
experiments/image_calibration/README.md
experiments/image_calibration/*.py
experiments/image_calibration/*.sh
experiments/image_calibration/*.json
experiments/image_calibration/*.jsonl
experiments/image_calibration/prompts/*.txt
papers/*.md
papers/*.txt
externals.lock.md
patches/reward-server/*.patch
```

Do not track generated artifacts, local environments, caches, raw benchmark outputs, nested third-party repo checkouts, `plan.md`, or `reports/` yet.

`externals.lock.md` records the URL and commit for `FastVideo/`, every direct repo under `evaluations/`, and the nested reward-server `mmdetection/` checkout. `patches/reward-server/*.patch` records the local reward-server and mmdetection compatibility changes without vendoring those repositories into root history.
