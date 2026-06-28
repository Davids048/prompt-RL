# DiffusionNFT Reproduction Plan: DrawBench First

## Summary

Set up a uv-managed reproduction environment in
`/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT`, using Python 3.12 and
latest CUDA 12.8 PyTorch instead of the README's conda/Python 3.10/CUDA 12.6
setup. Download the README reward checkpoints into the repo-local
`/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts`, install the
reward stack, start the UnifiedReward service in a separate uv venv, then run
only the DrawBench evaluation first with CFG.

## Experiment Contract

```text
Goal: Reproduce the provided DiffusionNFT LoRA checkpoint on DrawBench.
Setup: Main uv env at /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/.venv; reward ckpts at /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts; UnifiedReward env at /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/.venv-sglang.
Launch: torchrun --nproc_per_node=8 scripts/evaluation.py --lora_hf_path worstcoder/SD3.5M-DiffusionNFT-MultiReward --model_type sd3 --dataset drawbench --guidance_scale 4.5 --mixed_precision fp16 --save_images.
Result: /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/evaluation_output/drawbench_<YYYYmmdd_HHMMSS>/.
Class: setup + one real experiment.
```

## Setup Steps

1. Create the main venv and install CUDA 12.8 PyTorch:

   ```bash
   cd /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade torch torchvision torchaudio
   ```

2. Install the repo and core dependencies without letting `setup.py` downgrade
   Torch:

   ```bash
   uv pip install -e . --no-deps
   uv pip install transformers==4.40.0 accelerate==1.4.0 diffusers==0.33.1 peft==0.10.0
   uv pip install numpy==1.26.4 pandas==2.2.3 scipy==1.15.2 scikit-learn==1.6.1 scikit-image==0.25.2
   uv pip install albumentations==1.4.10 opencv-python==4.11.0.86 pillow==10.4.0 tqdm==4.67.1 wandb==0.18.7 pydantic==2.10.6 requests matplotlib==3.10.0
   uv pip install aiohttp==3.11.13 fastapi==0.115.11 uvicorn==0.34.0 huggingface-hub==0.29.1 datasets==3.3.2 tokenizers==0.19.1
   uv pip install deepspeed==0.16.4 bitsandbytes==0.45.3 xformers absl-py ml_collections sentencepiece einops==0.8.1 nvidia-ml-py==12.570.86
   ```

3. Build Flash Attention with the approved job cap:

   ```bash
   MAX_JOBS=16 uv pip install flash-attn==2.7.4.post1 --no-build-isolation
   ```

   Apply the same `MAX_JOBS<=16` rule in any other venv that builds
   `flash-attn`.

4. Download README reward checkpoints:

   ```bash
   mkdir -p /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts
   cd /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts
   wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
   wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
   wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
   wget https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
   ```

5. Install reward dependencies needed by DrawBench's scorer path:

   ```bash
   source /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/.venv/bin/activate
   cd /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT
   uv pip install open-clip-torch clip-benchmark hpsv2x==1.2.0 image-reward openai
   uv pip install "git+https://github.com/openai/CLIP.git"
   ```

   DrawBench uses `imagereward`, `pickscore`, `aesthetic`, `unifiedreward`,
   `clipscore`, and `hpsv2` in
   `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/scripts/evaluation.py`.

6. Set up and launch UnifiedReward separately:

   ```bash
   cd /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT
   uv venv --python 3.12 .venv-sglang
   source .venv-sglang/bin/activate
   uv pip install --index-url https://download.pytorch.org/whl/cu128 --upgrade torch
   uv pip install "sglang[all]"
   python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
   ```

## Preflight Validation

Run these checks before the real evaluation:

```bash
source /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/.venv/bin/activate
python - <<'PY'
import torch, diffusers, peft, transformers
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("peft", peft.__version__)
print("transformers", transformers.__version__)
PY
test -f /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts/sac+logos+ava1-l14-linearMSE.pth
test -f /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts/open_clip_pytorch_model.bin
test -f /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/reward_ckpts/HPS_v2.1_compressed.pt
curl -s http://127.0.0.1:17140/v1/models
```

## DrawBench Launch

Run only DrawBench first, using the CFG inference path:

```bash
source /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/.venv/bin/activate
cd /mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT

torchrun --nproc_per_node=8 scripts/evaluation.py \
  --lora_hf_path "worstcoder/SD3.5M-DiffusionNFT-MultiReward" \
  --model_type sd3 \
  --dataset drawbench \
  --guidance_scale 4.5 \
  --mixed_precision fp16 \
  --save_images
```

## Dataset Defaults

The script uses `DistributedSampler`, so actual generated counts can be padded
to a multiple of the process count when a dataset size is not divisible by 8.

| Dataset     | Prompt source                                                                              | Unique samples | Generated with 8 GPUs | Default output directory                                                                              |
| ----------- | ------------------------------------------------------------------------------------------ | -------------: | --------------------: | ----------------------------------------------------------------------------------------------------- |
| `drawbench` | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/dataset/drawbench/test.txt`          |          1,000 |                 1,000 | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/evaluation_output/drawbench_<YYYYmmdd_HHMMSS>/` |
| `pickscore` | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/dataset/pickscore/test.txt`          |          2,048 |                 2,048 | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/evaluation_output/pickscore_<YYYYmmdd_HHMMSS>/` |
| `geneval`   | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/dataset/geneval/test_metadata.jsonl` |          2,212 |                 2,216 | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/evaluation_output/geneval_<YYYYmmdd_HHMMSS>/`   |
| `ocr`       | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/dataset/ocr/test.txt`                |          1,018 |                 1,024 | `/mnt/weka/home/junda.su/codes/prompt-RL/DiffusionNFT/evaluation_output/ocr_<YYYYmmdd_HHMMSS>/`       |

The DrawBench output directory should contain:

- `evaluation_results.jsonl`
- `average_scores.json`
- `images/`, because `--save_images` is passed

## Assumptions

- Use `worstcoder/SD3.5M-DiffusionNFT-MultiReward` as the provided DiffusionNFT
  LoRA checkpoint because it is the model linked in the repo README.
- If PEFT loading rejects the checkpoint as non-LoRA, stop and report the
  checkpoint format mismatch instead of substituting another checkpoint.
- Do not run `pickscore`, `geneval`, or `ocr` evaluation until DrawBench
  completes and the result directory is inspected.
- GenEval and OCR-specific environment setup can be deferred until those
  benchmarks are actually requested; they are not needed for the first DrawBench
  run path.
- The real evaluation must run from a GPU-visible shell. The current planning
  shell previously could not communicate with the NVIDIA driver via `nvidia-smi`.
