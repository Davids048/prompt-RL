# External Repository Pins

This workspace vendors no third-party source trees in the root Git history. Clone these external repositories separately, then apply the local patches in `patches/reward-server/` where noted.

| Local path                                | Remote URL                                          | Branch   | Commit     | Local status                                      |
| ----------------------------------------- | --------------------------------------------------- | -------- | ---------- | ------------------------------------------------- |
| `FastVideo`                               | `https://github.com/hao-ai-lab/FastVideo.git`       | `main`   | `2c137931` | Clean                                             |
| `evaluations/CLIP`                        | `https://github.com/openai/CLIP.git`                | `main`   | `d05afc4`  | Clean                                             |
| `evaluations/HPSv2`                       | `https://github.com/tgxs002/HPSv2.git`              | `master` | `866735e`  | Clean                                             |
| `evaluations/HPSv3`                       | `https://github.com/MizzenAI/HPSv3.git`             | `main`   | `bd0c5fc`  | Clean                                             |
| `evaluations/PickScore`                   | `https://github.com/yuvalkirstain/PickScore.git`    | `main`   | `f79607f`  | Clean                                             |
| `evaluations/UnifiedReward`               | `https://github.com/CodeGoat24/UnifiedReward.git`   | `main`   | `9c64c2b`  | Clean                                             |
| `evaluations/align_sd`                    | `https://github.com/tgxs002/align_sd.git`           | `main`   | `78a49a3`  | Clean                                             |
| `evaluations/clipscore`                   | `https://github.com/jmhessel/clipscore.git`         | `main`   | `1036465`  | Clean                                             |
| `evaluations/flow_grpo`                   | `https://github.com/yifan123/flow_grpo.git`         | `main`   | `879042c`  | Clean                                             |
| `evaluations/geneval`                     | `https://github.com/djghosh13/geneval.git`          | `main`   | `af4902f`  | Clean                                             |
| `evaluations/reward-server`               | `https://github.com/yifan123/reward-server.git`     | `main`   | `8245af2`  | Apply `local-geneval-reward-server.patch`         |
| `evaluations/reward-server/mmdetection`   | `https://github.com/open-mmlab/mmdetection.git`     | `2.x`    | `e9cae2d`  | Apply `mmdetection-mmcv-version-guard.patch`      |

Notes:

- `FastVideo/` and `evaluations/` are intentionally ignored by the root repo to avoid vendoring nested Git checkouts.
- Runtime environments, model caches, generated media, and benchmark outputs are also intentionally ignored.
- `plan.md` and `reports/` are not tracked yet by current project choice.
