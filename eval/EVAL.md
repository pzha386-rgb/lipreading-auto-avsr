# Evaluation — Exact Reproduction Commands

Environment: BitaHub container, 1× NVIDIA A100-SXM4-80GB, PyTorch image.
All data under `/data/default/`.

## Dependencies (fresh container)

```bash
pip install soundfile pytorch-lightning editdistance sentencepiece av \
    --break-system-packages
```

## 1. LRS3 in-domain (1,321 utterances)

```bash
cd /data/default/auto_avsr

python eval.py \
  --modality video \
  --root-dir /data/default/lrs3_dataset \
  --test-file lrs3_test.csv \
  --pretrained-model-path /data/default/vsr_trlrs2lrs3vox2avsp_base.pth \
  2>&1 | tee /data/default/results/eval_strong_ckpt_$(date +%Y%m%d).txt
```

Result (2026-07-15 re-run, identical to first run):

```
wer = 0.2671385109424591    → 26.71 %
```

Runtime ≈ 17 min at ~1.28 it/s.

### Weaker checkpoint (for training-scale comparison)

Same command with `vsr_trlrs3vox2_base.pth` (1,759 h) → **31.15 %**.
Scaling training data 1,759 h → 3,291 h improves WER by 4.44 pp.

## 2. BBC out-of-domain (91 utterances, Tom Scott)

Test set construction: see `scripts/` (clip cutting → facexlib mouth ROI →
Auto-AVSR CSV manifest). Directory layout required by the dataloader:

```
bbc_test/
├── bbc/          ← symlink target dir (dataset name prefix in CSV)
├── rois/         ← 96×96 mouth ROI mp4s
└── labels/bbc_test.csv
```

```bash
cd /data/default/auto_avsr

python eval.py \
  --modality video \
  --root-dir /data/default/bbc_test \
  --test-file bbc_test.csv \
  --pretrained-model-path /data/default/vsr_trlrs2lrs3vox2avsp_base.pth \
  2>&1 | tee /data/default/bbc_test/eval_log_bbc_$(date +%Y%m%d).txt
```

Result:

```
wer = 0.2897125482559204    → 28.97 %
```

Runtime ≈ 2 min at ~0.79 it/s.

## 3. Interpretation

| | LRS3 (in-domain) | BBC (out-of-domain) | Δ |
|---|---:|---:|---:|
| WER | 26.71 % | 28.97 % | +2.26 pp |

The out-of-domain penalty is small, indicating the 3,291 h model generalises
well to unseen speakers/recording conditions, provided the same mouth-ROI
preprocessing is applied.

Gap to the paper's 20.3 % on LRS3 is attributed to dataset re-encoding: the
original LRS3 distribution is no longer available; the HuggingFace mirror we
used stores re-encoded mp4s (quality loss at the input to the visual
frontend). See design report, Discussion.
