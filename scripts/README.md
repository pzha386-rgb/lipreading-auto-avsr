# Data-preparation scripts (BBC test set)

These are the exact scripts that produced the BBC out-of-domain test set
(91 utterances, WER 28.97 %).

| File | Purpose |
|---|---|
| `extract_rois.py` | facexlib RetinaFace + AWing-FAN → 96×96 mouth-ROI mp4s |
| `convert_bbc_manifest.py` | build Auto-AVSR-format `labels/bbc_test.csv` (tokenised transcripts) |
| `bbc_test_example.csv` | resulting manifest (91 rows) for format reference |

Note: the initial clip-cutting step (raw video → utterance clips at 25 fps
mpeg4) was done with ad-hoc ffmpeg commands driven by a subtitle-based
segment list; the equivalent one-liner per clip is:

```bash
ffmpeg -y -i source.mp4 -ss <start> -to <end> -vf fps=25 -c:v mpeg4 -q:v 2 -an clips/clip_XXXX.mp4
```

Resulting BBC test-set layout consumed by `auto_avsr/eval.py`:

```
bbc_test/
├── bbc/                    ← dataset-name dir referenced by CSV prefix
├── clips/                  ← intermediate utterance clips
├── rois/clip_00xx.mp4      ← 91 × mouth-ROI videos (96×96)
└── labels/bbc_test.csv     ← "bbc,rois/clip_0000.mp4,<frames>,<token ids>"
```
