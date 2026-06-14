# Qwen3.5-9B Rationale

Last updated: 2026-06-13

Use `Qwen/Qwen3.5-9B` as the first prompt-enhancer model candidate for the RL
video-generation experiment.

Reasons:

1. The NEWTON paper is closely related to the intended experiment shape.
   It treats video generation as one action/tool inside a larger agentic loop,
   keeps the video generator unmodified, and trains the planner/prompt-side
   component on-policy with Flow-GRPO.

2. `Qwen/Qwen3.5-9B` is one of the newer Qwen model releases and is small enough
   for fast iteration before considering larger 100B+ models.

Local planning artifact:

```text
/home/hal-jundas/codes/UniRL/papers/rl_prompt_enhancer_experiment_plan.md
```

Source:

```text
arXiv:2605.18396
NEWTON: Agentic Planning for Physically Grounded Video Generation
Submitted 2026-05-18, revised 2026-05-19
https://arxiv.org/abs/2605.18396
```
