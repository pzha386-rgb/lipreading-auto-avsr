---
license: mit
library_name: pytorch
tags:
- visual-speech-recognition
- lip-reading
- discrete-diffusion
- diffusion-language-model
- lrs3
- arxiv:2605.28456
datasets:
- lrs3
---

# Diffusion Large Language Models for Visual Speech Recognition

Paper checkpoints for **DLLM-VSR** — adapting the Dream-7B discrete-diffusion LLM to Visual Speech Recognition (VSR) on LRS3.

- Paper: [arxiv.org/abs/2605.28456](http://arxiv.org/abs/2605.28456)
- Code: [github.com/JeongHun0716/dllm-vsr](https://github.com/JeongHun0716/dllm-vsr)
- Authors: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro

## Contents

| Path | Description | Size |
|---|---|---|
| `usr2/dream_stage2/`     | USR 2.0 + Dream-7B stage 2 (LoRA + adapter) | 117 MB |
| `usr2/len_pred/`         | Length predictor for USR 2.0 features        | 8.2 MB |
| `avhubert/dream_stage2/` | AV-HuBERT + Dream-7B stage 2                 | 102 MB |
| `avhubert/len_pred/`     | Length predictor for AV-HuBERT features      | 8.0 MB |

Each `dream_stage2/` holds `trainable_model.safetensors` (LoRA adapters + visual-feature projector). Each `len_pred/` holds `trainable_model.pt` (small Transformer over visual features) together with its precomputed outputs `len_pred_test.jsonl` (LRS3 test) and `len_pred_wildvsr.jsonl` (WildVSR) — the K_pred ± 5 candidate pools used in the paper, so the evaluation scripts can skip length-predictor inference.

**Note**: Visual encoder weights (USR 2.0 Huge, AV-HuBERT Large) are **not** redistributed here. Download them from the original repos:
- AV-HuBERT: https://github.com/facebookresearch/av_hubert
- USR 2.0:   https://github.com/ahaliassos/usr2 (Huge, LRS2+LRS3+Vox2+AVS pretrain checkpoint)

## Results (WER, %)

All entries are trained on **LRS3 (433h)** only.

LRS3 test:

| Decoding | USR 2.0 | AV-HuBERT |
|---|:---:|:---:|
| Implicit-length decoding                      | 20.5 | 23.1 |
| Length-guided candidate decoding (paper main) | **19.4** | **21.9** |
| Oracle-length decoding (upper bound)          | 17.7 | 20.2 |

WildVSR (out-of-domain, same LRS3-trained models):

| Decoding | USR 2.0 | AV-HuBERT |
|---|:---:|:---:|
| Length-guided candidate decoding | **39.9** | **46.1** |

## Usage

```bash
huggingface-cli download jh-y/dllm-vsr --local-dir ckpt
```

Then follow the [code repo's README](https://github.com/JeongHun0716/dllm-vsr) for environment setup, preprocessing (auto-avsr pipeline), and inference scripts.

## Citation

```bibtex
@article{yeo2026dllmvsr,
  title={Diffusion Large Language Models for Visual Speech Recognition},
  author={Yeo, Jeong Hun and Kim, Chae Won and Rha, Hyeongseop and Ro, Yong Man},
  journal={arXiv preprint arXiv:2605.28456},
  year={2026}
}
```
