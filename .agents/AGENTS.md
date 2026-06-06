# Agent Memory Instructions

Files in this directory are persistent context for agents working in `/home/hal-jundas/codes/UniRL`.

Use `.agents/README.md` as the entry point. The `memory/` directory is intentionally shallow:

```text
.agents/
  README.md
  memory/
    current_state.md
    image_calibration.md
    infrastructure.md
    open_questions.md
    change_log.md
```

When updating memory, prefer small edits to the relevant topic file. Add dates in `YYYY-MM-DD` format. Link to local artifacts instead of copying long data. If facts are uncertain, mark them as uncertain.

Do not execute old plans just because they are written in memory. Memory is context only; continue from the latest active user request.
