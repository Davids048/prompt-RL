# UniRL Experiment Hub

This workspace tracks the maintained UniRL experiment surface plus small
compatibility patches. Large third-party checkouts stay outside the root Git
history and are reconstructed from pinned commits.

The external pin list lives in `/home/hal-jundas/codes/UniRL/externals.lock.md`.

## Tracking Model

Root Git should track:

- maintained experiment code under `/home/hal-jundas/codes/UniRL/rl_prompt_enhancer/`;
- reusable prompt/data inputs under `/home/hal-jundas/codes/UniRL/data/`;
- research notes and local agent memory;
- small patch files such as
  `/home/hal-jundas/codes/UniRL/patches/slime/qwen35-prompt-enhancer-rl-compat.patch`.

Root Git should not track full external repositories such as
`/home/hal-jundas/codes/UniRL/slime` or
`/home/hal-jundas/codes/UniRL/Megatron-LM`.

## Slime Setup

Clone Slime at the pinned commit, then apply the UniRL compatibility patch:

```bash
cd /home/hal-jundas/codes/UniRL
git clone https://github.com/THUDM/slime.git slime
git -C slime checkout 5d7296a77a83bb249b257ee1f082d83db16a8079
git -C slime apply --check ../patches/slime/qwen35-prompt-enhancer-rl-compat.patch
git -C slime apply ../patches/slime/qwen35-prompt-enhancer-rl-compat.patch
```

The patch currently covers the Qwen3.5-9B local training/conversion path:

- disables Transformer Engine and fused-kernel-only options in the Qwen3.5-9B
  model script so the local Megatron path can run;
- makes HF config validation tolerate config fields that are absent from raw
  Qwen-style configs;
- keeps Qwen3.5 HF/Megatron bridge mappings compatible with the installed
  Transformers package;
- uses SGLang's `/model_info` endpoint for the weight version;
- handles BSHD attention without packed sequence params when GDN disables
  sequence packing;
- makes HF-to-torch-dist bridge selection fall back cleanly for Qwen3.5.

When the local Slime changes are intentional, refresh the tracked patch from
the Slime checkout:

```bash
cd /home/hal-jundas/codes/UniRL
git -C slime diff --binary --output=/home/hal-jundas/codes/UniRL/patches/slime/qwen35-prompt-enhancer-rl-compat.patch
git -C slime diff --check
```

## Megatron-LM Setup

Megatron-LM is reconstructed from the base commit and patch shipped by Slime.
Do not track Megatron-LM changes separately in root Git unless that decision is
reopened.

Slime pins the Megatron base commit in both
`/home/hal-jundas/codes/UniRL/slime/build_conda.sh` and
`/home/hal-jundas/codes/UniRL/slime/docker/Dockerfile`:

```text
1dcf0dafa884ad52ffb243625717a3471643e087
```

For a fresh checkout:

```bash
cd /home/hal-jundas/codes/UniRL
git clone --recursive https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
git -C Megatron-LM checkout 1dcf0dafa884ad52ffb243625717a3471643e087
git -C Megatron-LM apply --check ../slime/docker/patch/latest/megatron.patch
git -C Megatron-LM apply ../slime/docker/patch/latest/megatron.patch
```

Slime also ships SGLang patches under
`/home/hal-jundas/codes/UniRL/slime/docker/patch/`. If rebuilding the full
Slime environment from source, follow Slime's own `build_conda.sh` or Dockerfile
for the SGLang patch order. This root repo only tracks the UniRL-specific Slime
compatibility patch.
