# Visual Speech Recognition — Auto-AVSR Reproduction & Deployment

Reproduction and deployment of the [Auto-AVSR](https://github.com/mpc001/auto_avsr) visual speech recognition (lip-reading) system, including in-domain evaluation (LRS3), out-of-domain evaluation (BBC/Tom Scott), a real-time webcam demo, and a remote-inference GUI.

> Semester 1 project — University of Auckland
> Base model: `vsr_trlrs2lrs3vox2avsp_base.pth` (VSR, trained on 3,291 h: LRS2 + LRS3 + VoxCeleb2 + AVSpeech)

---

## Results Summary

| Experiment | Dataset | Samples | WER | Notes |
|---|---|---:|---:|---|
| In-domain eval | LRS3 test | 1,321 | **26.71 %** | Paper reports 20.3 %; gap analysed in report (video re-encoding loss on mirrored dataset) |
| In-domain eval (weaker ckpt) | LRS3 test | 1,321 | 31.15 % | `vsr_trlrs3vox2_base.pth`, 1,759 h |
| Out-of-domain eval | BBC (Tom Scott) | 91 | **28.97 %** | Only +2.26 pp vs in-domain → good generalisation |
| GUI end-to-end test | Single raw video | 1 | 10.53 % | Clear frontal face, full auto pipeline |

---

## Repository Layout

```
.
├── README.md                  ← you are here
├── scripts/                   ← data preparation (BBC test set construction)
│   ├── make_clips.py          ← cut raw video into utterance clips
│   ├── extract_rois.py        ← mouth ROI extraction (facexlib RetinaFace + AWing-FAN)
│   └── convert_bbc_manifest.py← build Auto-AVSR-format test CSV
├── eval/                      ← evaluation commands & logs
│   ├── EVAL.md                ← exact reproduction commands
│   └── logs/                  ← WER result logs
├── server/                    ← remote inference service (GPU host)
│   ├── server.py              ← FastAPI: /submit /status /infer /health
│   ├── roi_utils.py           ← server-side mouth ROI extraction
│   └── start.sh               ← one-shot launcher (server + cloudflare tunnel)
├── gui/                       ← local test GUI
│   └── gui.py                 ← Gradio frontend, async task polling
├── realtime/                  ← local webcam demo
│   └── realtime_lipreading.py ← mediapipe mouth tracking + live ROI preview
└── docs/                      ← design report & figures
```

---

## System Architecture

```
                 LOCAL (Windows)                      REMOTE (BitaHub, A100 GPU)
┌──────────────────────────────────┐        ┌───────────────────────────────────────┐
│  gui.py (Gradio, localhost:7860) │        │  server.py (FastAPI, :8000)           │
│   1. upload video                │  HTTPS │   POST /submit  → task_id             │
│   2. POST /submit ───────────────┼────────┼─→ background thread:                  │
│   3. poll GET /status/{id} ◄─────┼────────┼─   ffmpeg 25fps re-encode             │
│   4. show prediction + WER       │ cloud- │    → mouth ROI (facexlib pipeline)    │
│                                  │ flare  │    → Auto-AVSR VSR inference          │
│  realtime_lipreading.py          │ tunnel │    → WER vs ground truth (optional)   │
│   webcam → mediapipe → ROI       │        │  GET /status → {done, prediction,…}   │
└──────────────────────────────────┘        └───────────────────────────────────────┘
```

Design decisions:
- **Async task pattern** (`/submit` + `/status` polling) — cloudflare quick tunnels kill requests after ~100 s; ROI extraction of a 10 s video takes longer than that, so long-running work moves to a background thread and the GUI polls.
- **Server-side ROI extraction reuses the exact facexlib pipeline used for the BBC evaluation** — GUI results are directly comparable to offline eval numbers.
- **Deployment bottleneck is the visual frontend, not the model**: for a 9 s / 225-frame video, ROI extraction ≈ 25 s while VSR inference ≈ 1.9 s on an A100.

---

## Quick Start

### 0. Prerequisites

- The `auto_avsr` codebase cloned next to these files (see upstream repo)
- Checkpoint `vsr_trlrs2lrs3vox2avsp_base.pth` (~956 MB)
- GPU host: PyTorch + CUDA image; local machine: Anaconda (CPU is fine)

```bash
# GPU host
pip install fastapi uvicorn python-multipart pytorch-lightning \
            sentencepiece av editdistance facexlib scikit-image \
            --break-system-packages

# Local (Windows)
pip install gradio requests mediapipe opencv-python
```

> **Windows note**: always `set KMP_DUPLICATE_LIB_OK=TRUE` before running anything torch-related (OpenMP duplicate-runtime conflict in Anaconda).

### 1. LRS3 evaluation (GPU host)

```bash
cd auto_avsr
python eval.py \
  --modality video \
  --root-dir  /path/to/lrs3_dataset \
  --test-file lrs3_test.csv \
  --pretrained-model-path /path/to/vsr_trlrs2lrs3vox2avsp_base.pth
```

Note `--test-file` takes the bare filename — the dataloader prepends `labels/` internally. Passing `labels/lrs3_test.csv` produces a `labels/labels/...` path error.

### 2. BBC out-of-domain evaluation (GPU host)

```bash
# one-time data prep
python scripts/make_clips.py            # raw video → utterance clips
python scripts/extract_rois.py          # clips → 96×96 mouth ROI videos
python scripts/convert_bbc_manifest.py  # → labels/bbc_test.csv

# evaluate
cd auto_avsr
python eval.py \
  --modality video \
  --root-dir  /path/to/bbc_test \
  --test-file bbc_test.csv \
  --pretrained-model-path /path/to/vsr_trlrs2lrs3vox2avsp_base.pth
```

### 3. Remote inference service + GUI

```bash
# GPU host — starts FastAPI server and prints a trycloudflare URL
bash server/start.sh
```

```bash
# Local Windows
cd gui
set KMP_DUPLICATE_LIB_OK=TRUE
python gui.py
# open http://localhost:7860 and paste the trycloudflare URL
```

GUI workflow: upload any video with a visible face → tick **Extract mouth ROI on server** → Run Inference → prediction (+ WER if you provide the ground-truth transcript).

### 4. Real-time webcam demo (local)

```bash
cd realtime
set KMP_DUPLICATE_LIB_OK=TRUE
python realtime_lipreading.py --mode simple
```

Shows live camera feed with detected 96×96 mouth ROI (the model-input view) in the corner. `--mode full` additionally runs VSR inference on the buffered frames (requires the checkpoint locally; slow on CPU).

---

## Known Issues & Environment Notes

| Issue | Fix |
|---|---|
| `labels/labels/...csv` not found | pass bare filename to `--test-file` |
| `PyAV is not installed` | `pip install av` |
| `OMP: Error #15` on Windows | `set KMP_DUPLICATE_LIB_OK=TRUE` |
| `editdistance` build fails on Windows (needs MSVC) | only needed for eval on GPU host; GUI uses `torchaudio.functional.edit_distance` |
| ffmpeg `libx264` missing on host | pipeline uses `mpeg4` encoder |
| Tunnel kills long requests (~100 s) | async `/submit` + `/status` polling |
| Gradio upload widget shows browser language | set browser preferred language to English |

---

## Acknowledgements

- [Auto-AVSR](https://github.com/mpc001/auto_avsr) — Ma, Petridis, Pantic (ICASSP 2023): *Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels*
- [facexlib](https://github.com/xinntao/facexlib) — face detection / alignment models
- LRS3 dataset (TED/TEDx); BBC out-of-domain clips used for academic evaluation only
