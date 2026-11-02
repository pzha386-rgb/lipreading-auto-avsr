# -*- coding: utf-8 -*-
"""
prep_multispeaker_cache.py  -  E1 multi-speaker test set cache

Walks all 8 test speakers, reads their per-video manifest.csv for transcripts,
pairs each entry with the already-extracted ROI mp4, converts to .npy,
and writes a unified /root/autodl-tmp/cache/multispeaker_test/manifest.csv

utt_id format: {speaker}__{video_id}__{clip_id}   (speaker separable for E6)

Run:
  cd /root/autodl-tmp/vsr_bench
  # smoke test (first 5 clips per speaker)
  python prep_multispeaker_cache.py --limit-per-speaker 5
  # full
  python prep_multispeaker_cache.py
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE

CLIPS_ROOT = "/root/autodl-tmp/speaker_clips/test"
ROI_ROOT = "/root/autodl-tmp/speaker_roi/multi_speaker/multi_speaker_video_seg16s/test"
NAME = "multispeaker_test"


def collect_rows(limit_per_speaker=0):
    rows = []
    speakers = sorted(os.listdir(CLIPS_ROOT))
    for sp in speakers:
        sp_dir = os.path.join(CLIPS_ROOT, sp)
        if not os.path.isdir(sp_dir):
            continue
        videos = sorted(os.listdir(sp_dir))
        kept_per_sp = 0
        for vid in videos:
            mf = os.path.join(sp_dir, vid, "manifest.csv")
            if not os.path.exists(mf):
                continue
            with open(mf, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cid = r["clip_id"]
                    roi_mp4 = os.path.join(ROI_ROOT, sp, vid, f"{cid}.mp4")
                    if not os.path.exists(roi_mp4):
                        continue  # ROI extraction may have skipped this clip
                    rows.append({
                        "utt_id": f"{sp}__{vid}__{cid}",
                        "speaker": sp,
                        "video": vid,
                        "text": r["text"].strip(),
                        "roi_path": roi_mp4,
                    })
                    kept_per_sp += 1
                    if limit_per_speaker and kept_per_sp >= limit_per_speaker:
                        break
            if limit_per_speaker and kept_per_sp >= limit_per_speaker:
                break
        print(f"  {sp}: {kept_per_sp} clips")
    return rows


def convert_one(roi_mp4, out_npy):
    v = torchvision.io.read_video(roi_mp4, pts_unit="sec")[0].numpy()  # (T,H,W,3)
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in v]).astype(np.uint8)
    if g.shape[1:] != (96, 96):
        g = np.stack([cv2.resize(f, (96, 96)) for f in g])
    np.save(out_npy, g)
    return g.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=0,
                    help="smoke test: only first N clips per speaker")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    a = ap.parse_args()

    out_dir = os.path.join(CACHE, NAME)
    os.makedirs(out_dir, exist_ok=True)

    print(f"collecting rows (limit_per_speaker={a.limit_per_speaker}) ...")
    rows = collect_rows(a.limit_per_speaker)
    print(f"total {len(rows)} clips\n")

    ok, bad = [], []
    for i, r in enumerate(rows):
        npy_name = f"{r['utt_id']}.npy"
        out_npy = os.path.join(out_dir, npy_name)
        if a.skip_existing and os.path.exists(out_npy):
            ok.append({"utt_id": r["utt_id"], "text": r["text"],
                       "npy": npy_name, "speaker": r["speaker"],
                       "video": r["video"]})
            continue
        try:
            shape = convert_one(r["roi_path"], out_npy)
            ok.append({"utt_id": r["utt_id"], "text": r["text"],
                       "npy": npy_name, "speaker": r["speaker"],
                       "video": r["video"]})
            if (i + 1) % 50 == 0 or i < 3:
                print(f"[{i+1}/{len(rows)}] {r['utt_id']} shape={shape}")
        except Exception as e:
            bad.append(r["utt_id"])
            print(f"[{i+1}/{len(rows)}] {r['utt_id']} FAILED: {e}")

    mf_path = os.path.join(out_dir, "manifest.csv")
    with open(mf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy", "speaker", "video"])
        w.writeheader()
        w.writerows(ok)
    print(f"\nDONE: {len(ok)} ok, {len(bad)} failed")
    print(f"manifest -> {mf_path}")
    if bad:
        print("failed ids:", bad[:20], "..." if len(bad) > 20 else "")


if __name__ == "__main__":
    main()
