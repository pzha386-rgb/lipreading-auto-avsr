# Upstream code used in this project

The model code is not copied into this repository. Clone each repository from its original source:

| Name in this project | Upstream repository | Used for |
|---|---|---|
| Auto-AVSR (T1, S0, S1, S6) | https://github.com/mpc001/auto_avsr | Supervised VSR models |
| VSR-ML (S2, S4, S5) | https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages | Ma et al. 2022 models |
| RAVEn / BRAVEn (T3, T4, R*, B*) | https://github.com/ahaliassos/raven | Self-supervised VSR models |
| USR 2.0 (U*) | [TBC: add link] | Semi-supervised unified models |
| AV-HuBERT (T2) | https://github.com/facebookresearch/av_hubert | Self-supervised model (visual-only mode) |
| AV-HuBERT MuAViC checkpoint | https://huggingface.co/nguyenvulebinh/AV-HuBERT-MuAViC-en | Checkpoint used for T2 |
| DLLM-VSR (L2, L3) | [TBC: add link] | Tested but not used in the thesis |
| facexlib | https://github.com/xinntao/facexlib | Face detection and landmarks |

Local changes to upstream code (for example `patched_batch_beam_search.py`) are kept in `thesis_export/vsr_bench/`.
