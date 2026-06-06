# Open Questions And Next Decisions

Last updated: 2026-05-28

The current pilot proves that the pipeline can run, but it is not yet a faithful PromptRL reproduction.

Key open points:

1. Full GenEval should use all 553 prompts from `evaluations/geneval/prompts/evaluation_metadata.jsonl`, not the 12-prompt pilot subset.
2. PromptRL's paper clearly reports GenEval, OCR, PickScore, HPS, and UnifiedReward, but the exact prompt pools for PickScore/HPS/UnifiedReward need verification from released code or prompt files before claiming a faithful reproduction.
3. OCR evaluation has not been run yet. The likely local source is `evaluations/flow_grpo`, but it needs extraction into a standalone runner.
4. UnifiedReward parsing should be fixed before scaling. The likely fix is to use the simpler official pointwise prompt from the UnifiedReward repo and rerun preference scoring.
5. The rewrite prompt is a major experimental knob. Even the conservative prompt added semantic details in the pilot, so future runs should version rewrite instructions explicitly.
6. For video experiments, select a small but representative T2V prompt suite and use native video evaluators first. Keyframe PickScore/HPS can be secondary diagnostics, not primary video metrics.

Suggested next image step:

```text
Run full 553-prompt GenEval with:
  none
  qwen25_vl_3b

Keep the same SD3.5 Medium settings:
  1024 resolution
  20 steps
  guidance 6.0
  fixed seeds

Report per-group GenEval scores and average.
```

Suggested evaluator cleanup before broader scoring:

```text
Patch UnifiedReward prompt/parsing.
Add a small OCR runner.
Verify PromptRL prompt/eval pools from released code if available.
```
