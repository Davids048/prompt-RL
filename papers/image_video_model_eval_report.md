# Image and Video Model Evaluation Papers

Generated: 2026-06-05

Reference notes:

- `/home/hal-jundas/codes/UniRL/papers/evals.md`
- `/home/hal-jundas/codes/UniRL/papers/related_works.md`
- `/home/hal-jundas/codes/UniRL/papers/dancegrpo_evaluation_findings.txt`

Scope:

- This report focuses on evaluation protocols, benchmarks, reward models, and prompt sets used by the papers mentioned in the scratch notes.
- It keeps the emphasis from the notes: image generation evals, image editing evals, and video generation evals for RL/post-training papers.
- I treat benchmark/reward papers such as GenEval, PickScore, HPS, VBench, VideoReward, and VisionReward as first-class entries because the method papers depend on them.

Important caveats:

- The checked MixGRPO paper at arXiv:2507.21802 is primarily a text-to-image/FLUX evaluation paper. In the Notion plan, keep the video/HPSv3 mention as a short HunyuanVideo1.5 comparison note rather than a separate MixGRPO video-eval entry.
- The checked DenseGRPO paper at arXiv:2601.20218 is image-focused in the available abstract/search record. I did not find native video evaluation for it.
- Several current papers are recent preprints. I preserved paper-reported metric names and roles, but exact leaderboards can change when code, benchmark versions, or reward checkpoints change.

## Executive Summary

- Image RL papers converge on a three-part evaluation stack:
  - Object and compositional correctness: GenEval, DrawBench, T2I-CompBench.
  - Text rendering: OCR accuracy, OCR1k-style prompt sets, MARIOEval and its LAION/TMDB/OpenLibrary subsets.
  - Human preference or model preference: PickScore, HPS/HPSv2/HPSv3, ImageReward, UnifiedReward, aesthetic score, and sometimes CLIPScore.
- Image editing papers add edit-specific rewards and task coverage:
  - PromptRL reports EditReward for FLUX.1-Kontext image editing.
  - OmniEdit uses an internal multi-task editing test set and human evaluation, with emphasis on seven editing task families and variable aspect ratios.
- Video papers should be tracked separately from image papers:
  - VBench measures broad video generation quality across disentangled dimensions.
  - VideoReward/VideoAlign scores video quality, motion quality, and text alignment.
  - VisionReward is used as a newer video reward/evaluator for visual quality, temporal consistency, and dynamic degree.
  - Human preference remains important because automated video rewards still miss temporal intent, physical plausibility, and subjective motion quality.
- For your Notion wiki, the clean split is:
  - Image evals: GenEval, DrawBench, T2I-CompBench, OCR/OCR1k, MARIOEval, PickScore, HPS, HPSv3, ImageReward, UnifiedReward, EditReward.
  - Video evals: VBench, VideoReward/VideoAlign, VisionReward, VidProM prompts, VideoJAM-bench, MotionBench, human pairwise preference.

## Evaluation Taxonomy

### Image Generation Evals

- GenEval:
  - What it tests: object existence, attribute binding, counting, position, color, and compositional prompt following.
  - How it is usually used: generate images for the official prompt metadata and score detections/object constraints.
  - Why it matters here: PromptRL, Flow-GRPO, DiffusionNFT, DenseGRPO, DanceGRPO, and UniRL-Zero-style image RL experiments report or reference it.
  - Source: https://arxiv.org/abs/2310.11513

- DrawBench:
  - What it tests: a curated prompt suite for hard text-to-image behavior, including color, counting, spatial relations, text, rare words, misspellings, and complex prompts.
  - How it is usually used: human preference, model comparison, and qualitative stress testing.
  - Why it matters here: PromptRL and Flow-GRPO notes use it as an aesthetic or broad T2I prompt suite, often alongside reward-model metrics.
  - Source: https://proceedings.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf

- T2I-CompBench / T2I-CompBench++:
  - What it tests: compositional text-to-image generation, including attribute binding, object relationships, numeracy, 3D spatial relationships, and complex compositions.
  - How it is usually used: category-level scoring for prompt-image alignment and compositional reasoning.
  - Why it matters here: PromptRL scratch notes mention it next to DrawBench for image-generation evaluation.
  - Source: https://arxiv.org/abs/2307.06350

- OCR / OCR1k:
  - What it tests: whether rendered text in generated images is recognizable and matches the requested text.
  - How it is usually used: OCR engine reads generated image text; score is usually accuracy, word-level match, or related precision/recall/F-measure.
  - Why it matters here: PromptRL reports OCR accuracy, Flow-GRPO reports OCR improvements, and the local workspace already uses Flow-GRPO OCR prompt conventions.

- MARIOEval:
  - What it tests: text-rich image generation and text rendering quality.
  - Subsets from TextDiffuser notes: LAIONEval4000, TMDBEval500, and OpenLibraryEval500, with additional MARIOEval subsets in later versions.
  - Metrics: FID, CLIPScore, OCR evaluation with accuracy/precision/recall/F-measure, and human evaluation.
  - Why it matters here: the scratch notes mention TMDB/OpenLib/MARIOEval as text-rendering eval references.
  - Source: https://arxiv.org/abs/2305.10855

- PickScore / Pick-a-Pic:
  - What it tests: predicted human preference between generated images for a prompt.
  - How it is usually used: reward model or offline evaluation metric for text-image preference alignment.
  - Why it matters here: PromptRL, Flow-GRPO, DiffusionNFT, MixGRPO, DenseGRPO, and local image calibration notes all treat PickScore as a core image preference metric.
  - Source: https://arxiv.org/abs/2305.01569

- HPS / HPSv2 / HPSv3:
  - What it tests: human-preference score for text-to-image outputs.
  - How it is usually used: model-preference metric, RL reward, or post-training diagnostic.
  - Why it matters here: DanceGRPO uses HPS-v2.1 for T2I, MixGRPO uses HPS-v2.1, and HPSv3 is a newer preference model referenced in the scratch notes as an image/video-relevant diagnostic.
  - HPSv2 source: https://arxiv.org/abs/2306.09341
  - HPSv3 source: https://arxiv.org/abs/2508.03789

- UnifiedReward:
  - What it tests: multimodal reward modeling for image-text alignment and preference-like quality.
  - How it is usually used: offline image reward metric and sometimes RL reward.
  - Why it matters here: Flow-GRPO, DiffusionNFT, MixGRPO, and local image calibration plans include UnifiedReward.
  - Source: https://arxiv.org/abs/2503.05236

- ImageReward and Aesthetic Score:
  - What they test: image-text human preference and visual appeal.
  - How they are usually used: auxiliary preference/aesthetic metrics, especially for DrawBench or broader T2I prompt suites.
  - Why they matter here: Flow-GRPO and DiffusionNFT report them alongside GenEval/OCR/PickScore/HPS-style scores.

### Image Editing Evals

- EditReward:
  - What it tests: preference/quality for instruction-guided image editing.
  - How it is used in these notes: PromptRL reports EditReward for FLUX.1-Kontext, including improvement over the base editing model.
  - Main caution: it is edit-specific, so it should not be mixed with plain T2I scores without a separate modality/task label.

- OmniEdit test set:
  - What it tests: broad instruction-guided editing coverage across seven task families, with different aspect ratios.
  - How OmniEdit evaluates: automatic evaluation plus human evaluation on a curated test set.
  - Why it matters here: PromptRL scratch notes mention OmniEdit-like editing evaluation and the PromptRL paper validates RL on image editing models.
  - Source: https://arxiv.org/abs/2411.07199

### Video Generation Evals

- VBench:
  - What it tests: broad video generation quality across multiple disentangled dimensions.
  - Dimensions emphasized by later papers: aesthetic quality, imaging quality, subject consistency, background consistency, motion smoothness, dynamic degree, and text alignment.
  - Why it matters here: VBench is the main native video benchmark in DenseDPO, Flow-DPO/VideoReward, and the scratch-note conclusion.
  - Source: https://arxiv.org/abs/2311.17982
  - Project: https://github.com/Vchitect/VBench

- VideoReward / VideoAlign:
  - What it tests: video quality, motion quality, and text alignment.
  - Common shorthand: VQ, MQ, and TA.
  - How it is used: reward model, evaluation score, and win-rate comparator for T2V models and RL methods.
  - Why it matters here: Flow-DPO introduces/uses this stack; DanceGRPO uses VideoAlign dimensions; the scratch notes highlight VideoAlign as a core video eval.
  - Source: https://arxiv.org/abs/2501.13918
  - Project: https://github.com/KwaiVGI/VideoAlign

- VisionReward:
  - What it tests: multimodal reward for visual generation, used in video papers for quality and temporal aspects.
  - How it appears in these notes: DanceGRPO and DenseDPO use VisionReward-style video metrics, especially visual quality, temporal consistency, and dynamic degree.
  - Source: https://arxiv.org/abs/2412.21059

- VidProM prompts:
  - What it is: prompt set used in DanceGRPO's T2V evaluation.
  - How it is used: T2V generation prompts for HunyuanVideo evaluation with VideoAlign and VisionReward-Video metrics.

- VideoJAM-bench and MotionBench:
  - What they test: challenging motion prompts and broad motion-focused video generation behavior.
  - How they are used: DenseDPO uses them to stress motion quality and compare against DPO baselines.

## Paper and Evaluation Source Notes

The entries below separate two roles:

- Method/framework papers: papers that use evaluation suites, reward models, or prompt sets to evaluate image/video generation methods.
- Eval/reward/prompt sources: papers or projects that introduce a benchmark, reward model, preference model, or prompt suite used by later method papers.

### PromptRL: Prompt Matters in RL for Flow-Based Image Generation

Source: https://arxiv.org/abs/2602.01382

- Role:
  - Method paper that uses existing image-generation and image-editing evals.
- Modality:
  - Image generation.
  - Image editing.
- What the scratch notes were pointing at:
  - PromptRL is important because it shows that prompt refinement/prompt enhancement changes RL training behavior and final evaluation scores.
  - The notes correctly connect PromptRL to GenEval, OCR, PickScore, HPS, editing rewards, and prompt-suite evaluations such as DrawBench/T2I-CompBench.
  - The notes also flag that PromptRL's composition, aesthetics, and text-rendering tasks mostly follow Flow-GRPO conventions, while image editing is the extra task family not covered by Flow-GRPO.
- Training/eval source notes from the scratch file:
  - Composition and text-rendering training are noted as using Flow-GRPO training data/prompt conventions.
  - Image editing training is noted as using 10k OmniEdit training examples with edit instruction plus reference image.
  - Image editing evaluation is noted as OmniEdit validation set/EditReward-style evaluation.
- Main reported evals:
  - GenEval:
    - Used for object-centric and compositional T2I correctness.
    - Paper abstract reports a GenEval score of 0.97.
  - OCR accuracy:
    - Used for text rendering.
    - Paper abstract reports OCR accuracy of 0.98.
  - PickScore:
    - Used as an image preference metric.
    - Paper abstract reports PickScore of 24.05.
  - HPS:
    - Used as a human-preference-style image quality/alignment metric in released result tables.
  - EditReward:
    - Used for image editing evaluation on FLUX.1-Kontext.
    - Paper abstract reports EditReward improvement from 1.19 to 1.43 with 0.06M rollouts.
- Related prompt/eval suites from the scratch notes:
  - DrawBench:
    - Used as a hard prompt suite for aesthetic and general T2I comparison.
  - T2I-CompBench:
    - Relevant for compositional prompt following.
  - MARIOEval/TMDB/OpenLibrary:
    - Relevant family for text rendering, although PromptRL's headline number is reported as OCR accuracy/OCR-style evaluation.
  - OmniEdit:
    - Relevant because PromptRL validates the RL idea on image editing, not only T2I.
- Practical takeaway:
  - Do not record PromptRL as "just another image RL paper." The eval story is specifically about whether a trainable prompt-refinement agent improves both score and generalization.
  - In Notion, keep separate rows for `PromptRL - T2I evals` and `PromptRL - editing evals`, or use a multi-select task field with `image_generation` and `image_editing`.

### UniRL-Zero: Reinforcement Learning on Unified Models with Joint Language Model and Diffusion Model Experts

Source: https://arxiv.org/abs/2510.17937

- Role:
  - Framework/method paper that uses generation rewards and eval scenarios.
- Modality:
  - Unified understanding and generation.
  - Image generation is one of the relevant generation scenarios.
- What the scratch notes were pointing at:
  - UniRL-Zero is relevant because it frames RL over unified language/diffusion experts, not just a standalone diffusion model.
  - The notes mention six RL scenarios and specific image-generation reward tasks.
- Main eval/reward scenarios visible from notes and paper summary:
  - Image reward toy/objectives:
    - JPEG compressibility.
    - JPEG incompressibility.
  - T2I RL with chain-of-thought style reasoning:
    - Notes cite GenEval improvement across baseline and CoT-style variants.
  - Unified model scenarios:
    - The paper defines six scenarios covering understanding, generation, and interaction between model experts.
- Evaluation role:
  - It is more of a framework/baseline paper than a benchmark paper.
  - For your wiki, record it as a "unified RL framework" whose generation evals should be mapped back to GenEval and simple image reward objectives.
- Practical takeaway:
  - Keep UniRL-Zero separate from pure image/video RL papers because its eval design is organized by unified model capability scenario.
  - In a Notion eval matrix, tag it with `unified_model`, `image_generation`, and `framework_baseline`.

### Imagen / DrawBench

Source: https://proceedings.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf

- Role:
  - Eval/prompt-suite source.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - DrawBench is a benchmark/prompt suite, not an RL method.
  - It appears in later papers because it provides hard prompts and human-comparison style evaluation.
- Eval details:
  - Prompt count:
    - 200 prompts.
  - Prompt categories:
    - color.
    - counting.
    - spatial relations.
    - text rendering.
    - rare words.
    - misspellings.
    - complex prompts.
    - other hard compositional or linguistic cases.
  - Evaluation style:
    - Primarily human preference and side-by-side model comparison in Imagen.
    - Later RL papers reuse DrawBench prompts with automatic reward models such as aesthetic score, ImageReward, UnifiedReward, or DeQA-style evaluators.
- Practical takeaway:
  - Treat DrawBench as a prompt-suite row in Notion, not as a metric by itself.
  - Link each paper using it to the downstream metric actually reported.

### T2I-CompBench / T2I-CompBench++ (Less important)

Source: https://arxiv.org/abs/2307.06350

- Role:
  - Eval/benchmark source.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - It is a compositional T2I benchmark referenced by PromptRL-style evaluations.
- Eval details:
  - T2I-CompBench:
    - 6,000 compositional prompts.
    - Main categories: attribute binding, object relationships, and complex compositions.
    - Subcategories: color binding, shape binding, texture binding, spatial relationships, non-spatial relationships, and complex compositions.
  - T2I-CompBench++:
    - 8,000 prompts.
    - Adds/expands categories such as generative numeracy and 3D spatial relationships.
    - Uses detection-based metrics and MLLM-based evaluation for compositional challenges.
- Practical takeaway:
  - Use T2I-CompBench when a paper claims compositional alignment beyond GenEval.
  - In Notion, put it under `Image evals -> compositionality`, separate from `Image evals -> object correctness` for GenEval.

### TextDiffuser / MARIOEval

Source: https://arxiv.org/abs/2305.10855

- Role:
  - Eval/benchmark source for text-rendering evaluation.
- Modality:
  - Image generation, focused on text rendering.
- What the scratch notes were pointing at:
  - The notes mention OCR1k, TMDB, OpenLib, and MARIOEval. These belong to the broader text-rendering evaluation family.
- Eval details:
  - MARIOEval:
    - Built to evaluate text rendering in generated images.
    - Includes LAIONEval4000, TMDBEval500, and OpenLibraryEval500 in the paper notes.
  - Metrics:
    - FID for generated image distribution quality.
    - CLIPScore for image-text similarity.
    - OCR evaluation for detected/recognized text.
    - Human evaluation for text rendering quality.
  - OCR metrics:
    - Accuracy.
    - Precision.
    - Recall.
    - F-measure.
- Practical takeaway:
  - For your image eval wiki, group OCR1k and MARIOEval under `text_rendering`.
  - Record whether a paper reports exact OCR accuracy, word-level F1, or a broader OCR-derived score, because these are not interchangeable.

### PickScore / Pick-a-Pic

Source: https://arxiv.org/abs/2305.01569

- Role:
  - Preference/reward-model source.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - PickScore is one of the repeated preference metrics used by the RL papers.
- Eval details:
  - PickScore is trained from human preference data for text-to-image generations.
  - It estimates which image a human would prefer for a given prompt.
  - It is useful for broad preference alignment, but it is not a direct test of object counting, rendered text accuracy, or video motion.
- Papers in these notes using or referencing it:
  - PromptRL.
  - Flow-GRPO.
  - DiffusionNFT.
  - MixGRPO.
  - DenseGRPO.
  - DanceGRPO T2I evaluation uses Pick-a-Pic/PickScore-style scoring.
- Practical takeaway:
  - In Notion, tag PickScore as `preference_model`, not `compositional_correctness`.

### HPS / HPSv2 / HPSv3

Sources:

- HPSv2: https://arxiv.org/abs/2306.09341
- HPSv3: https://arxiv.org/abs/2508.03789

- Role:
  - Preference/reward-model source.
- Modality:
  - Primarily image generation preference.
  - HPSv3 appears in the scratch notes as a newer preference metric to track where papers explicitly use it.
- What the scratch notes were pointing at:
  - HPS appears repeatedly as a preference score in RL papers.
  - The scratch-note conclusion calls out HPSv3 as one of the metrics to track.
- Eval details:
  - HPS/HPSv2:
    - Human preference style score for generated images conditioned on prompts.
    - Often reported as HPS-v2.1 in T2I evaluations.
  - HPSv3:
    - Newer preference model/version.
    - Useful to track separately because scores are not necessarily comparable with HPSv2.
- Papers in these notes using or referencing it:
  - PromptRL.
  - DanceGRPO T2I.
  - DiffusionNFT.
  - MixGRPO.
  - DenseGRPO.
- Practical takeaway:
  - Keep HPSv2 and HPSv3 as separate Notion eval records.
  - Record the exact HPS version a paper reports.

### UnifiedReward

Source: https://arxiv.org/abs/2503.05236

- Role:
  - Multimodal reward-model source.
- Modality:
  - Image generation.
  - General multimodal reward modeling.
- What the scratch notes were pointing at:
  - UnifiedReward appears as a stronger reward/evaluator option alongside ImageReward, PickScore, and HPS.
- Eval details:
  - Used as an automatic preference/reward score.
  - Useful when a paper wants a single multimodal reward estimate beyond CLIPScore or aesthetics.
  - It should be interpreted as a learned reward model, not as a ground-truth human study.
- Papers in these notes using or referencing it:
  - Flow-GRPO.
  - DiffusionNFT.
  - MixGRPO.
  - Local image calibration notes.
- Practical takeaway:
  - In Notion, classify it as `reward_model` and `image_preference`.
  - Add a note that local evaluation already has a UnifiedReward checkout/server path in `/home/hal-jundas/codes/UniRL/evaluations/UnifiedReward`.

### GenEval

Source: https://arxiv.org/abs/2310.11513

- Role:
  - Eval/benchmark source.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - GenEval is the clearest repeated image correctness benchmark across the RL papers.
- Eval details:
  - Tests whether requested objects and relations appear in the generated image.
  - Common categories include single object, two objects, counting, colors, position, and color attribution.
  - Uses structured prompts/metadata and object-detection style scoring.
- Papers in these notes using or referencing it:
  - PromptRL.
  - Flow-GRPO.
  - DiffusionNFT.
  - DanceGRPO T2I.
  - DenseGRPO.
  - UniRL-Zero T2I-RL examples.
- Practical takeaway:
  - In Notion, make GenEval the central `image_correctness` eval entry.
  - Add a local implementation note: official prompt metadata exists at `/home/hal-jundas/codes/UniRL/evaluations/geneval/prompts/evaluation_metadata.jsonl`.

### Flow-GRPO: Training Flow Matching Models via Online RL

Source: https://arxiv.org/abs/2505.05470

- Role:
  - Method paper that uses multiple image evals and provides useful local prompt/reward conventions.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - Flow-GRPO is a key baseline for RL fine-tuning of flow/diffusion image models.
  - The notes emphasize that it uses multiple automatic image evals, not just one reward.
- Main evals:
  - GenEval:
    - Used for compositional/object correctness.
    - Reported as a major improvement for SD3.5-M in search/paper snippets.
  - OCR:
    - Used for text rendering.
    - Reported as another major improvement target.
    - Scratch-note formula: `r = max(1 - Ne / Nref, 0)`, where `Ne` is the minimum edit distance between rendered text and target text, and `Nref` is the number of target characters inside quotation marks in the prompt.
    - The same rule-based reward is used as a text-accuracy metric in the noted Flow-GRPO setup.
  - PickScore:
    - Used for preference alignment.
    - Some results include KL regularization variants.
  - DrawBench:
    - Used as a broad prompt suite.
    - Metrics attached to DrawBench include aesthetic score, DeQA, ImageReward, and UnifiedReward in paper notes/search snippets.
  - CLIPScore/HPS/ImageReward/UnifiedReward:
    - Used as auxiliary image-text or preference metrics depending on experiment section.
- Practical takeaway:
  - Flow-GRPO is the baseline row that connects the local workspace's image calibration eval stack to published RL-for-generation papers.
  - For Notion, link Flow-GRPO to GenEval, OCR, PickScore, DrawBench, ImageReward, UnifiedReward, and HPS-style diagnostics.

### DiffusionNFT

Sources:

- Paper: https://arxiv.org/abs/2509.16117
- Project/code: https://github.com/NVlabs/DiffusionNFT

- Role:
  - Method paper that uses image-generation evals and reward metrics.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - DiffusionNFT is another image RL/post-training paper that should be mapped against the same eval stack as Flow-GRPO.
- Main evals from paper/project notes:
  - GenEval.
  - OCR.
  - PickScore.
  - DrawBench.
  - CLIPScore.
  - HPSv2.1.
  - Aesthetic score.
  - ImageReward.
  - UnifiedReward.
- Reported emphasis:
  - The project/search snippets emphasize sample efficiency and strong GenEval improvement.
  - It compares against Flow-GRPO-style baselines.
- Practical takeaway:
  - In Notion, put DiffusionNFT next to Flow-GRPO under `image_rl`.
  - Its eval mapping is broad: correctness, text rendering, preference, aesthetics, and prompt-suite diagnostics.

### DanceGRPO

Sources:

- Paper: https://arxiv.org/abs/2505.07818
- Local findings: `/home/hal-jundas/codes/UniRL/papers/dancegrpo_evaluation_findings.txt`

- Role:
  - Method paper that uses both image and video evals.
- Modality:
  - Image generation.
  - Text-to-video generation.
  - Image-to-video generation.
- What the scratch notes were pointing at:
  - DanceGRPO is useful because it spans image and video, and it shows how papers separate T2I, T2V, and I2V evaluation stacks.
- T2I evals:
  - Models:
    - SD1.4.
    - FLUX.1-dev / FLUX.
    - HunyuanVideo-T2I.
  - Metrics:
    - HPS-v2.1.
    - CLIP Score.
    - Pick-a-Pic / PickScore.
    - GenEval.
  - Prompt/eval protocol from local findings:
    - 1000 test prompts for CLIP/Pick-a-Pic style evaluation.
    - Official prompts for GenEval and HPS-v2.1.
- T2V evals:
  - Model:
    - HunyuanVideo.
  - Prompt source:
    - VidProM prompts.
  - Metrics:
    - VideoAlign video quality (VQ).
    - VideoAlign motion quality (MQ).
    - VideoAlign text alignment (TA).
    - VisionReward-Video in reported tables.
  - Caution:
    - Local findings note that final analysis excludes unstable TA in some discussion while tables still report TA and VisionReward-Video.
- I2V evals:
  - Model:
    - SkyReels-I2V.
  - Reference images:
    - Synthesized by HunyuanVideo-T2I.
    - Scratch notes separately mention `ConsisID + synth img (FLUX)` as the training prompt/reference-image source; local detailed findings point to HunyuanVideo-T2I references, so this should be verified before writing it into Notion as a fixed fact.
  - Metric:
    - VideoAlign motion quality.
- Human evaluation:
  - 240 T2I prompts.
  - 200 T2V prompts.
  - 200 I2V prompts plus reference images.
- Practical takeaway:
  - In Notion, DanceGRPO should be one paper record with three evaluation sections: `T2I`, `T2V`, and `I2V`.
  - This is a good template for separating image and video eval taxonomies in the wiki.

### VBench

Sources:

- Paper: https://arxiv.org/abs/2311.17982
- Project: https://github.com/Vchitect/VBench

- Role:
  - Video eval/benchmark source.
- Modality:
  - Video generation.
- What the scratch notes were pointing at:
  - VBench is the core native video benchmark.
- Eval details:
  - Designed as a comprehensive benchmark suite with disentangled dimensions.
  - Includes prompt suite, generated videos, and human preference annotations.
  - Common dimensions used by downstream papers:
    - Aesthetic quality.
    - Imaging quality.
    - Subject consistency.
    - Background consistency.
    - Motion smoothness.
    - Dynamic degree.
    - Text alignment.
- Papers in these notes using or referencing it:
  - DenseDPO.
  - Flow-DPO/VideoReward.
  - Scratch-note conclusion for video evals.
- Practical takeaway:
  - In Notion, VBench should be the anchor entry for `video_benchmark`.
  - When a paper reports only VBench frame/visual dimensions but not motion/text alignment, capture the exact dimensions used.

### DenseDPO

Source: https://arxiv.org/abs/2506.03517

- Role:
  - Method paper that uses video evals and human evaluation.
- Modality:
  - Video generation.
- What the scratch notes were pointing at:
  - DenseDPO is a video preference/post-training paper whose evals are native video metrics plus human study.
- Main benchmark/prompt sets:
  - VideoJAM-bench:
    - 128 challenging motion prompts.
  - MotionBench:
    - 419 motion-focused prompts.
    - Includes prompts from sets such as MovieGenBench.
- Automatic evals:
  - VBench:
    - Aesthetic quality.
    - Imaging quality.
    - Subject consistency.
    - Background consistency.
    - Motion smoothness.
    - Dynamic degree.
    - Text alignment.
  - VisionReward:
    - Visual quality.
    - Temporal consistency.
    - Dynamic degree.
- Human evaluation:
  - Compares DenseDPO against StructuralDPO and VanillaDPO.
  - Dimensions:
    - Text alignment.
    - Visual quality.
    - Temporal consistency.
    - Dynamic degree.
- Practical takeaway:
  - DenseDPO belongs in the Notion video section, linked to VBench and VisionReward.
  - Record the VBench and VisionReward dimensions it reports.

### VisionReward

Source: https://arxiv.org/abs/2412.21059

- Role:
  - Reward/eval source used by downstream video papers.
- Modality:
  - Visual generation reward, with use in video evaluation.
- What the scratch notes were pointing at:
  - The scratch-note conclusion includes VisionReward as a video eval/reward to track.
- Eval role:
  - Reward/evaluator for visual outputs.
  - In video papers, it appears as a model-based measure for visual quality, temporal consistency, and dynamic degree.
- Papers in these notes using or referencing it:
  - DanceGRPO.
  - DenseDPO.
- Practical takeaway:
  - In Notion, make VisionReward a `reward_model` entry linked to video papers, but annotate the exact variant used, such as VisionReward-Video when papers distinguish it.

### Flow-DPO / VideoReward / VideoAlign

Sources:

- Paper: https://arxiv.org/abs/2501.13918
- Project: https://github.com/KwaiVGI/VideoAlign

- Role:
  - Method paper that also introduces the VideoReward/VideoAlign evaluation stack.
- Modality:
  - Video generation.
- What the scratch notes were pointing at:
  - This is the main source for the VideoReward/VideoAlign eval stack.
- VideoReward details:
  - Built around three core dimensions:
    - Video quality (VQ).
    - Motion quality (MQ).
    - Text alignment (TA).
  - Uses a Qwen2-VL-2B style backbone in the reported notes.
  - Trained/evaluated with a BTT-style loss in the paper notes.
- Reward-model evals:
  - VideoGen-RewardBench.
  - GenAI-Bench.
- Generator evals:
  - VideoReward score.
  - VideoReward win rate.
  - VBench.
- Prompt sets:
  - VBench prompts.
  - VideoGen-Eval.
  - TA-Hard.
- Methods compared:
  - Flow-DPO.
  - Flow-RWR.
  - Flow-NRG.
- Practical takeaway:
  - In Notion, split `VideoReward` as a metric/reward entry and `Flow-DPO` as a method paper.
  - Link Flow-DPO to VideoReward/VideoAlign and VBench.

### MixGRPO

Source: https://arxiv.org/abs/2507.21802

- Role:
  - Method paper that uses image-generation rewards/evals.
- Modality:
  - Checked source: image generation, especially FLUX.1-dev T2I.
  - The scratch-note video/HPSv3 mention should be treated as a short HunyuanVideo1.5 comparison note, not as a separate MixGRPO video-eval version.
- What the scratch notes were pointing at:
  - The notes list MixGRPO near video evals, but the available checked paper centers on image RL and reward mixing.
- Main checked evals:
  - HPDv2 prompt split:
    - 103,700 training prompts.
    - 400 test prompts.
  - HPS-v2.1.
  - PickScore.
  - ImageReward.
  - UnifiedReward.
  - CLIPScore in some ablations.
  - Efficiency:
    - Number of function evaluations (NFE).
    - Wall-clock time or training cost comparisons.
- Practical takeaway:
  - In Notion, mark MixGRPO as `image_rl`.
  - Add only a short note for the HunyuanVideo1.5 comparison; do not create a separate video-eval entry for MixGRPO.

### DenseGRPO

Source: https://arxiv.org/abs/2601.20218

- Role:
  - Method paper that uses image-generation rewards/evals.
- Modality:
  - Image generation.
- What the scratch notes were pointing at:
  - DenseGRPO appears in the same family as Flow-GRPO/DiffusionNFT: image RL with dense rewards or dense credit assignment.
- Main checked evals:
  - GenEval.
  - OCR.
  - PickScore.
  - Aesthetic score.
  - ImageReward.
  - HPS/PickScore-style preference metrics in related comparisons.
- Reported emphasis:
  - Search snippets compare DenseGRPO against Flow-GRPO with stronger GenEval/OCR/PickScore results.
  - The available checked source does not establish native video evaluation.
- Practical takeaway:
  - In Notion, place DenseGRPO under `image_rl`.
  - Link it to GenEval, OCR, PickScore, and ImageReward/Aesthetic diagnostics.

### OmniEdit

Source: https://arxiv.org/abs/2411.07199

- Role:
  - Image-editing dataset/model/eval source.
- Modality:
  - Image editing.
- What the scratch notes were pointing at:
  - PromptRL's editing-side evaluation mentions OmniEdit-style editing tasks/datasets.
  - OmniEdit itself is a useful image-editing benchmark/model reference.
- Eval details:
  - Test set:
    - Curated images with different aspect ratios.
    - Diverse instructions covering multiple editing tasks.
  - Task coverage:
    - Seven different image editing task families.
  - Evaluation:
    - Automatic evaluation.
    - Human evaluation.
  - Data-quality note:
    - OmniEdit argues that CLIP-score filtering alone is weak, and uses multimodal model scoring such as GPT-4o for data-quality oriented sampling.
- Practical takeaway:
  - In Notion, tag OmniEdit as `image_editing`, not plain T2I.
  - Include OmniEdit as the scratch-note source for PromptRL's editing train/validation setup; do not add a separate verification task for now.

## Paper-to-Eval Mapping

### Eval, Reward, and Prompt Sources

- GenEval:
  - Role: introduces an image-generation compositional/object correctness benchmark.
  - Used by: PromptRL, Flow-GRPO, DiffusionNFT, DanceGRPO T2I, DenseGRPO, and UniRL-Zero-style T2I examples.
- Imagen / DrawBench:
  - Role: introduces a hard text-to-image prompt suite and human-eval setup.
  - Used by: Flow-GRPO and related image-RL evaluation notes for prompt-suite testing.
- TextDiffuser / MARIOEval:
  - Role: introduces text-rendering evaluation sets and OCR-style metrics.
  - Used by: the scratch-note text-rendering benchmark family, including TMDB/OpenLibrary-style prompts.
- PickScore / Pick-a-Pic:
  - Role: introduces an image preference reward/evaluator.
  - Used by: PromptRL, Flow-GRPO, DiffusionNFT, MixGRPO, DenseGRPO, and DanceGRPO T2I.
- HPSv2 / HPSv3:
  - Role: introduce human-preference-style image reward/eval models.
  - Used by: PromptRL, DanceGRPO T2I, DiffusionNFT, MixGRPO, DenseGRPO, and the scratch-note HPSv3 reference.
- UnifiedReward:
  - Role: introduces a multimodal reward model.
  - Used by: Flow-GRPO, DiffusionNFT, MixGRPO, and local image-calibration notes.
- OmniEdit:
  - Role: introduces an image-editing dataset/model/eval source.
  - Used by: PromptRL editing notes as the image-editing train/validation source.
- VBench:
  - Role: introduces a broad video-generation benchmark.
  - Used by: DenseDPO and Flow-DPO/VideoReward.
- VisionReward:
  - Role: introduces a visual reward/evaluator used by downstream video papers.
  - Used by: DanceGRPO and DenseDPO.
- VideoReward / VideoAlign:
  - Role: introduced with Flow-DPO as a video reward/eval stack.
  - Used by: DanceGRPO T2V/I2V and Flow-DPO-style video generation evaluation.

### Method and Framework Papers That Use Evals

- PromptRL:
  - Image generation: GenEval, OCR/OCR1k, PickScore, HPS.
  - Image editing: EditReward, FLUX.1-Kontext, OmniEdit train/validation source from the scratch notes.
- UniRL-Zero:
  - Unified RL scenarios and image-generation examples such as JPEG compressibility/incompressibility and GenEval-style T2I-RL.
  - Supporting context in this report; not part of the original scratch-note reading-list update.
- Flow-GRPO:
  - Image generation RL using GenEval, OCR, PickScore, DrawBench, aesthetic score, DeQA, ImageReward, and UnifiedReward.
- DiffusionNFT:
  - Image generation RL using GenEval, OCR, PickScore, DrawBench, CLIPScore, HPSv2.1, aesthetic score, ImageReward, and UnifiedReward.
- DanceGRPO:
  - T2I: HPS-v2.1, CLIP Score, Pick-a-Pic/PickScore, and GenEval.
  - T2V: VidProM prompts, VideoAlign VQ/MQ/TA, and VisionReward-Video.
  - I2V: SkyReels-I2V with reference images and VideoAlign MQ.
  - Human eval across T2I/T2V/I2V.
- DenseDPO:
  - Video generation using VideoJAM-bench, MotionBench, VBench, VisionReward, and human evaluation.
- Flow-DPO:
  - Video generation preference optimization using VideoReward/VideoAlign, VideoGen-RewardBench, GenAI-Bench, and VBench.
- MixGRPO:
  - Checked source: image generation.
  - Uses HPS-v2.1, PickScore, ImageReward, UnifiedReward, CLIPScore, NFE/time.
  - Scratch-note HPSv3/video relation should be kept as a short comparison note only, not as a separate video-eval version of MixGRPO.
- DenseGRPO:
  - Image generation using GenEval, OCR, PickScore, aesthetic score, and ImageReward.

## Suggested Notion Wiki Plan

Do not execute this yet. This plan keeps the Notion work limited to paper reading-list entries plus an append-only update to the existing eval summary page.

### 1. Reading List Updates in Task Hub

- Search the existing reading list first.
- If a paper already has an entry, update that entry instead of creating a duplicate.
- If a paper from the original scratch note is missing, create a reading-list entry for it.
- Mark only the papers/eval sources mentioned in `/home/hal-jundas/codes/UniRL/papers/evals.md` as `Read`.
- Use the reading list's existing properties only, follow existing conventions. 
- Put extra eval details in the page body or in the target eval-summary page, not as new reading-list database properties.
- Do not add a local filesystem link to this markdown report in Notion.

Reading-list entries to create or update and mark as `Read`:

- PromptRL.
- Imagen / DrawBench.
- PickScore / Pick-a-Pic.
- HPSv2.
- HPSv3.
- UnifiedReward.
- GenEval.
- Flow-GRPO.
- TextDiffuser / MARIOEval.
- OmniEdit.
- DanceGRPO.
- DiffusionNFT.
- VBench.
- DenseDPO.
- VisionReward.
- Flow-DPO / VideoReward / VideoAlign.
- MixGRPO.
- DenseGRPO.

Supporting entries from this report that should not be marked `Read`:

- UniRL-Zero.
- T2I-CompBench / T2I-CompBench++.

### 2. Append to the Eval Summary Page

Target page:

- https://www.notion.so/Image-Video-Model-Eval-37551add94548144967ee5aa3b02a696?source=copy_link

Update rule:

- Append new content to the page.
- Do not rewrite, delete, or reorganize existing page content.
- Do not create separate benchmark/eval pages unless they already exist and the user asks for cross-linking later.

Append these sections:

- `Image Generation Evals`
  - GenEval.
  - DrawBench.
  - OCR/OCR1k.
  - MARIOEval.
  - PickScore.
  - HPS/HPSv2/HPSv3.
  - ImageReward.
  - UnifiedReward.
  - Aesthetic Score.
- `Image Editing Evals`
  - EditReward.
  - OmniEdit train/validation source from the scratch notes.
  - Human editing preference where a paper reports it.
- `Video Generation Evals - Reward Used During Training or Alignment`
  - VideoAlign / VideoReward.
  - HPSv3 where explicitly reported by the paper being summarized.
- `Video Generation Evals - VBench`
  - VBench overall score.
  - VBench dimensions when the paper reports them.
- `Video Generation Evals - Other Automatic Metrics`
  - VisionReward / VisionReward-Video.
  - VideoJAM-bench.
  - MotionBench.
  - VidProM prompts.
- `Video Generation Evals - Human Preference`
  - Human pairwise or side-by-side preference studies.
  - Dimensions reported by the paper, such as visual quality, text alignment, temporal consistency, dynamic degree, T2I/T2V/I2V preference, or editing preference.

### 3. Plain Paper-to-Eval Matrix for the Summary Page

Append a plain table like this to the target Notion page:

| Entry                         | Role                  | Modality        | Eval / reward items                                      | Notion note                                      |
| ----------------------------- | --------------------- | --------------- | -------------------------------------------------------- | ------------------------------------------------ |
| PromptRL                      | Uses evals            | Image + editing | GenEval, OCR, PickScore, HPS, EditReward                 | Include OmniEdit train/validation note.          |
| Flow-GRPO                     | Uses evals            | Image           | GenEval, OCR, PickScore, DrawBench, ImageReward          | Include OCR edit-distance reward formula.        |
| DiffusionNFT                  | Uses evals            | Image           | GenEval, OCR, PickScore, HPSv2.1, UnifiedReward          | Compare as image-RL eval stack.                  |
| DanceGRPO                     | Uses evals            | Image + video   | GenEval, HPSv2.1, PickScore, VideoAlign, VisionReward    | Split notes into T2I, T2V, and I2V.              |
| DenseDPO                      | Uses evals            | Video           | VBench, VisionReward, VideoJAM-bench, MotionBench        | Include human-eval dimensions.                   |
| Flow-DPO / VideoReward        | Introduces + uses     | Video           | VideoReward, VideoAlign, VBench                          | Treat VideoReward as eval stack.                 |
| MixGRPO                       | Uses evals            | Image           | HPS-v2.1, PickScore, ImageReward, UnifiedReward          | Add only short HunyuanVideo1.5 comparison note.  |
| DenseGRPO                     | Uses evals            | Image           | GenEval, OCR, PickScore, Aesthetic, ImageReward          | Note that scratch file says no video eval.       |
| GenEval                       | Introduces eval       | Image           | Object/compositional correctness                         | Eval source, not a method paper.                 |
| Imagen / DrawBench            | Introduces prompts    | Image           | DrawBench prompts and human eval                         | Prompt suite source.                             |
| PickScore / Pick-a-Pic        | Introduces reward     | Image           | Image preference score                                   | Reward/eval source.                              |
| HPSv2 / HPSv3                 | Introduces reward     | Image           | Human-preference score                                   | Record exact version used by each paper.         |
| UnifiedReward                 | Introduces reward     | Image           | Multimodal reward                                        | Reward/eval source.                              |
| TextDiffuser / MARIOEval      | Introduces eval       | Image text      | OCR, FID, CLIPScore, human eval                          | Text-rendering eval source.                      |
| OmniEdit                      | Introduces data/eval  | Image editing   | Editing tasks, train/validation source, human eval       | Editing source, not plain T2I.                   |
| VBench                        | Introduces eval       | Video           | Video benchmark dimensions                               | Video eval source.                               |
| VisionReward                  | Introduces reward     | Video / visual  | Visual quality, temporal consistency, dynamic degree     | Reward/eval source.                              |

### 4. Minimal Page Creation Rules

- Create or update only paper reading-list entries.
- Append the structured summary and matrix to the existing eval summary page.
- Do not create separate pages for every benchmark, reward model, or eval suite.
- If a benchmark/reward paper already has a reading-list entry, use that existing entry.
- Keep cross-linking light: link method-paper entries to the summary page if useful, but do not build a separate wiki graph yet.
