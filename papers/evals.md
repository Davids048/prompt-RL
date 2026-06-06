promptrl
tasks: composition -- aesthetics -- text rendering (OCR) -- image edit (not in flowgrpo)
    - tasks mostly follow flow grpo 

Geneval composition
GenEval validation set for compositional evaluation,

> training: 
- trained on training set from flow GRPO 

-------

aesthetics 
https://proceedings.neurips.cc/paper_files/paper/2022/file/ec795aeadae0b7d230fa35cbaf04c041-Paper-Conference.pdf
We introduce DrawBench, a new structured suite of text prompts for text-to-image evaluation. DrawBench enables deeper insights through a multi-dimensional evaluation of text-to-image models, with
text prompts designed to probe different semantic properties of models. These include compositionality, cardinality, spatial relations, the ability to handle complex text prompts or prompts with
rare words, and they include creative prompts that push the limits of models’ ability to generate
highly implausible scenes well beyond the scope of the training data. With DrawBench, extensive
human evaluation shows that Imagen outperforms other recent methods [59, 12, 56] by a significant
margin. We further demonstrate some of the clear advantages of the use of large pre-trained language
models [54] over multi-modal embeddings such as CLIP [51] as a text encoder for Imagen.
DrawBench, a set of challenging and compositional prompts

DrawBench contains 11 categories of prompts, testing
different capabilities of models such as the ability to faithfully render different colors, numbers of
objects, spatial relations, text in the scene, and unusual interactions between objects. Categories
also include complex prompts, including long, intricate textual descriptions, rare words, and also
misspelled prompts. We also include sets of prompts collected from DALL-E [55], Gary Marcus et
al. [40] and Reddit. Across these 11 categories, DrawBench comprises 200 prompts in total, striking
a good balance between the desire for a large, comprehensive dataset, and small enough that human
evaluation remains feasible. (Appendix C provides a more detailed description of DrawBench. Fig. 2
shows example prompts from DrawBench with Imagen samples.)

metrics used with draw bench: 
PickScore  (model output score) 
https://github.com/yuvalkirstain/pickscore
pickscore- model trained based on clip 
Pick-a-Pic, a large, open dataset of text-to-image prompts and real users’ preferences over generated images
https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1
https://huggingface.co/yuvalkirstain/PickScore_v1


HPS (model output score)
https://arxiv.org/pdf/2306.09341
https://github.com/tgxs002/HPSv2
https://arxiv.org/abs/2508.03789
https://mizzenai.github.io/HPSv3.project/
https://github.com/MizzenAI/HPSv3


unified reward 
https://arxiv.org/abs/2503.05236


In flow GRPO: (Fetch the paper) 
"""
to detect reward hacking beyond task-specific accuracy, we
evaluate four automatic image quality metrics: Aesthetic Score [59], DeQA [60], ImageReward [32],
and UnifiedReward [61] (see Appendix B.1 for details). All metrics are computed on DrawBench [1],
a comprehensive benchmark with diverse prompts for T2I models.
"""

trianing: 
trained on pick a pic dataset 


-------------

text rendering benchmarks -- rule based, acc. 
In flowGRPO:
"""
    Following [58],
we measure text fidelity with the reward r = max(1 − Ne/Nref, 0), where Ne is the minimum edit
distance between the rendered text and the target text and Nref is the number of characters inside the
quotation marks in the prompt. This reward also serves as our metric of text accuracy.
"""


ocr-1k val set from flowgrpo  (prompts) -- score: 
TMDB,
OpenLib from MARIOEval

training: 
trained on training set from flow grpo 

---

image edit 
OmniEdit validation set

training:
- trained on 10k examples from omni edit training set w/ edit instruction + ref img. 




========================
Dance GRPO 
Tasks:
- T2I, T2V, I2V

T2I 
Reward model: HPSv2.1, CLIP Score (both with binary reward)
Eval:
    1000 test prompts - metrics: clip scores, pick a pic performance (sec.3.2)
    GenEval, HPSv2.1 official benchmarks 

T2V
reward: videoalign 
prompts (Train): vidprom 
Eval:
    1000 test prompts -- metric: videoalign score 

I2V 
reward: videoalign 
Prompt(train): consisID + synth img (FLUX) 
Eval:
    - 1000 test prompts -- metric video align 

NOTE: T2V, I2V mostly shown graph of how video align metric (reward) grows


=====================================
diffusion NFT
GenEval, OCR
Pickscore, CLIP score, HPSv2.1, Aesthetics, ImgReward, Unireward 

========================

VBench

============

DenseDPO
https://arxiv.org/pdf/2506.03517
VideoJamBench: VBench + Visionreward metrics

MotionBench: VBench + visionreward metrics 

Note: VIsion reward https://arxiv.org/pdf/2412.21059, Not video reward 

=================
FLOW-DPO
https://arxiv.org/pdf/2501.13918
"""
We then introduce VideoReward, a multi-dimensional
video reward model, and examine how annotations and various design choices
impact its rewarding efficacy.
"""
"""
we introduce three alignment
algorithms for flow-based models. These include two training-time strategies:
direct preference optimization for flow (Flow-DPO) and reward weighted regression for flow (Flow-RWR), and an inference-time technique, Flow-NRG, which
applies reward guidance directly to noisy videos.
"""
"""
We use Qwen2-VL-2B [72] as the backbone of our reward model, trained with
BTT loss. """

Eval for reward model (Not the generator model)
VideoGen-RewardBench
    - build on VideoGen-Eval (T2V) 
GenAI-Bench
    - short vid 2s
    - Eval on early generation outputs 

Eval For Video gen
- Video reward score 
- VBench score 


============
MixGRPO
https://arxiv.org/pdf/2507.21802
Video align score + HPSv3

some others for T2I 



============
DENSEGRPO
https://arxiv.org/pdf/2601.20218
Does not contain video eval 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Conclusion: 
For Imagegen: 
- GenEval 
- Text Rendering 
- Model based preference 

For Video 
- VideoAlign, HPSv3 rewards
- VBench score 
- * VisionReward (only seen in DenseDPO)




