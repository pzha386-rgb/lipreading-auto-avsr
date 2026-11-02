# Lip Motion to Text: A Reliable Lip-Reading Module for Robots

Master's research project (45 points), University of Auckland, 2026.
Visual-only speech recognition (VSR): benchmark of public models, confidence and calibration, and new-speaker adaptation.

The Semester 1 README (Auto-AVSR reproduction, GUI, real-time demo) is in `docs/README_semester1.md`.

## Project stages

| Stage | What | Main result |
|---|---|---|
| Benchmark | 29 model configurations (Auto-AVSR, VSR-ML, RAVEn, BRAVEn, USR 2.0) on LRS3 and a BBC set of 91 clips, with 8 video perturbations | `thesis_export/figures/table_main.csv` |
| Case Study A | One new speaker (Tom Scott, 302 min): confidence, LLM correction, fine-tuning, LoRA, calibration, ROVER, full system | see table below |
| Case Study B | 25 YouTube speakers (17 train / 8 test, 570 min): baseline, group LoRA, personal LoRA, rejection | see table below |

## Where to find each result

| Experiment | Files |
|---|---|
| Model list and settings | `thesis_export/vsr_bench/models.csv` |
| Benchmark WER, speed, memory | `thesis_export/figures/table_main.csv`, `thesis_export/results_v2/*.meta.json` |
| Robustness (blur, low-res, dark, frame drop) | `thesis_export/figures/stats_robustness.csv`, `fig6_robustness.png` |
| Bootstrap confidence intervals | `thesis_export/figures/stats_ci.csv`, `stats_pairs.csv` |
| Error types and confusions | `thesis_export/figures/error_*.csv` |
| Semester 1 four-model results (raw vs normalised WER) | `thesis_export/results/*_full_summary.txt` |
| Confidence scores per clip | `thesis_export/day1/conf/*_confA.csv` |
| Confidence ranking, AURC, tiers | `thesis_export/day1/figs/*_conf_summary.csv`, `tiers_summary.csv` |
| LLM correction (Qwen2.5-7B) | `thesis_export/day1/figs/llm_summary_*.csv` |
| Full fine-tuning (A5, A15, A30) | `thesis_export/logs/adapt_all.log`, `adapt_v2.log` |
| LoRA (L5 to Lall) | `thesis_export/logs/lora_all.log`, `thesis_export/adapt_lora/*/history.json` |
| Multi-speaker baseline (E1) | `thesis_export/logs/E1_*.log`, `thesis_export/results/E1_*_by_speaker.csv` |
| Group LoRA (E2) | `thesis_export/logs/E2*.log` |
| Personal adaptation (E3) | `thesis_export/logs/E3_*.log` |
| Rejection sweep (E8) | `thesis_export/demo/e8_results/` |
| Curated per-speaker evaluation and data audit | `thesis_export/demo/` |

## Main scripts

| Script | Purpose |
|---|---|
| `vsr_bench/eval_autoavsr_nbest.py`, `eval_braven_nbest.py` | Decode with N-best list and save confidence scores |
| `vsr_bench/lora_adapt.py` | LoRA adaptation (rank 8, 1.18 M trainable parameters) |
| `vsr_bench/adapt_autoavsr.py` | Full / partial fine-tuning |
| `vsr_bench/calibrate_threshold.py` | Few-shot threshold calibration |
| `vsr_bench/rover_combine.py` | Word-level ROVER of two models |
| `vsr_bench/full_system_eval.py` | LoRA + calibration + rejection |
| `vsr_bench/reliability_diagram.py` | Reliability diagram and ECE |
| `vsr_bench/perturb_rois.py` | Video perturbations for robustness tests |
| `vsr_bench/stats_bootstrap.py` | Bootstrap confidence intervals |
| `demo/e8_posthoc.py` | Coverage sweep and ROVER on the multi-speaker test set |
| `vsr_bench/gui/lip_compare_gui.py` | Demo GUI: compare models on a clip, key findings, live webcam test |
| `vsr_bench/lipread_service.py`, `start_services.sh` | Model services used by the GUI (Auto-AVSR, BRAVEn, USR 2.0) |

## Demo GUI

The GUI (Gradio front-end, FastAPI model services on a cloud GPU) has three tabs.

| Tab | Screenshot |
|---|---|
| Watch a clip: all models on the same LRS3 or BBC clip, wrong words marked | `docs/images/gui/watch_clip_long.jpg` |
| Key findings: label efficiency, LRS3 vs BBC, speed, robustness | `docs/images/gui/findings_*.jpg` |
| Live test: webcam recording, three models, time per stage | `docs/images/gui/live_test_webcam.jpg` |

![Live test](docs/images/gui/live_test_webcam.jpg)

## Semester 1 code (root folders)

| Folder | Content |
|---|---|
| `eval/` | Semester 1 evaluation notes and logs (`EVAL.md`) |
| `gui/`, `server/` | First GUI and model server |
| `realtime/` | Real-time lip-reading demo |
| `scripts/` | Mouth ROI extraction and BBC manifest tools |
| `docs/` | Semester 1 README, design report, images |

## Data

Videos, mouth ROIs and model checkpoints are not in this repository (size and copyright).
The lists of large files are in `thesis_export/_large_files.txt`.
Upstream code is listed in `UPSTREAM.md`.
