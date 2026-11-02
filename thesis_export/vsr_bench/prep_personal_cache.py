# -*- coding: utf-8 -*-
"""
prep_personal_cache.py  -  E3 personal adaptation cache

For one test speaker, split their clips per video by time (first N% adapt,
rest eval), and build two caches that drop into the existing
adapt_autoavsr.py + eval_autoavsr_nbest.py pipeline:

  /root/autodl-tmp/cache/personal_<speaker>_adapt/
    (npy symlinks + manifest_frames.csv + manifest.csv + split.json)
  /root/autodl-tmp/cache/personal_<speaker>_eval/
    (npy symlinks + manifest.csv)

The adapt cache uses PSEUDO video_ids (one per clip) so adapt_autoavsr.py's
video-level train/val split can carve a small validation subset out of the
adapt clips for early stopping.

Run:
  python prep_personal_cache.py --speaker jack_edwards
  python prep_personal_cache.py --speaker jack_edwards --adapt-ratio 0.3 --val-frac 0.1
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE

SRC_CACHE = os.path.join(CACHE, "multispeaker_test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--adapt-ratio", type=float, default=0.3,
                    help="first fraction of each video's clips go to adapt")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="within adapt set, fraction reserved for early-stop val")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # 1. load source manifest
    src_mf = os.path.join(SRC_CACHE, "manifest.csv")
    with open(src_mf, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["speaker"] == a.speaker]
    if not rows:
        raise SystemExit(f"no clips found for speaker '{a.speaker}' in {src_mf}")

    # 2. group by video, sort by clip_id (time order)
    by_vid = {}
    for r in rows:
        by_vid.setdefault(r["video"], []).append(r)
    for v in by_vid:
        by_vid[v].sort(key=lambda r: r["utt_id"])  # clip_0000, clip_0001, ...

    # 3. split per video
    adapt_rows, eval_rows = [], []
    for v, items in sorted(by_vid.items()):
        n = len(items)
        k = max(1, int(round(n * a.adapt_ratio)))
        adapt_rows.extend(items[:k])
        eval_rows.extend(items[k:])

    # 4. frames for each clip (adapt_autoavsr needs this)
    def frames_of(npy_name):
        return int(np.load(os.path.join(SRC_CACHE, npy_name), mmap_mode="r").shape[0])

    print(f"\n=== speaker: {a.speaker} ===")
    print(f"videos: {len(by_vid)}   total clips: {len(rows)}")
    print(f"adapt: {len(adapt_rows)} clips   eval: {len(eval_rows)} clips")

    # 5. ADAPT cache
    adapt_dir = os.path.join(CACHE, f"personal_{a.speaker}_adapt")
    os.makedirs(adapt_dir, exist_ok=True)
    adapt_mf_frames = []
    pseudo_vids_all = []
    for i, r in enumerate(adapt_rows):
        pv = f"pseudo_{i:04d}"
        pseudo_vids_all.append(pv)
        # symlink npy
        src_npy = os.path.join(SRC_CACHE, r["npy"])
        dst_npy = os.path.join(adapt_dir, r["npy"])
        if not os.path.exists(dst_npy):
            os.symlink(src_npy, dst_npy)
        adapt_mf_frames.append({
            "utt_id": r["utt_id"],
            "clip_id": r["utt_id"].split("__")[-1],
            "video_id": pv,
            "speaker": a.speaker,
            "path": r["npy"],
            "npy": r["npy"],
            "text": r["text"],
            "frames": frames_of(r["npy"]),
        })

    with open(os.path.join(adapt_dir, "manifest_frames.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "clip_id", "video_id", "speaker",
                                           "path", "npy", "text", "frames"])
        w.writeheader(); w.writerows(adapt_mf_frames)

    with open(os.path.join(adapt_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy", "speaker", "video"])
        w.writeheader()
        for r in adapt_mf_frames:
            w.writerow({"utt_id": r["utt_id"], "text": r["text"], "npy": r["npy"],
                        "speaker": r["speaker"], "video": r["video_id"]})

    # 6. split.json: 90% pseudo_vids train, 10% val
    rng = random.Random(a.seed)
    shuffled = list(pseudo_vids_all); rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * a.val_frac)))
    val_pvs = sorted(shuffled[:n_val])
    train_pvs = sorted(shuffled[n_val:])
    split_path = os.path.join(CACHE, f"personal_{a.speaker}_adapt_split.json")
    with open(split_path, "w") as f:
        json.dump({"train": train_pvs, "val": val_pvs}, f, indent=2)

    adapt_min = sum(r["frames"] for r in adapt_mf_frames) / 25 / 60
    train_min = sum(r["frames"] for r in adapt_mf_frames if r["video_id"] in train_pvs) / 25 / 60
    val_min = sum(r["frames"] for r in adapt_mf_frames if r["video_id"] in val_pvs) / 25 / 60

    print(f"\n[ADAPT CACHE]  {adapt_dir}")
    print(f"  total adapt: {len(adapt_mf_frames)} clips, {adapt_min:.1f} min")
    print(f"  train: {len(train_pvs)} clips, {train_min:.1f} min   "
          f"val: {len(val_pvs)} clips, {val_min:.1f} min")

    # 7. EVAL cache
    eval_dir = os.path.join(CACHE, f"personal_{a.speaker}_eval")
    os.makedirs(eval_dir, exist_ok=True)
    eval_mf = []
    for r in eval_rows:
        src_npy = os.path.join(SRC_CACHE, r["npy"])
        dst_npy = os.path.join(eval_dir, r["npy"])
        if not os.path.exists(dst_npy):
            os.symlink(src_npy, dst_npy)
        eval_mf.append({
            "utt_id": r["utt_id"], "text": r["text"], "npy": r["npy"],
            "speaker": r["speaker"], "video": r["video"],
        })
    with open(os.path.join(eval_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy", "speaker", "video"])
        w.writeheader(); w.writerows(eval_mf)

    eval_min = sum(frames_of(r["npy"]) for r in eval_rows) / 25 / 60
    print(f"\n[EVAL CACHE]   {eval_dir}")
    print(f"  {len(eval_mf)} clips, {eval_min:.1f} min")
    print(f"\nUse:")
    print(f"  adapt --name personal_{a.speaker}_adapt")
    print(f"  eval  --dataset personal_{a.speaker}_eval")


if __name__ == "__main__":
    main()
