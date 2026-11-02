# -*- coding: utf-8 -*-
"""
prep_multispeaker_train_cache.py

Build the training cache that adapt_autoavsr.py expects:
  /root/autodl-tmp/cache/multispeaker_train/
    <speaker>__<video>__<clip>.npy
    manifest_frames.csv        <-- has: clip_id, video_id, speaker, path, npy, text, frames
    manifest.csv               <-- for compatibility with eval scripts (utt_id, text, npy)

Also writes:
  /root/autodl-tmp/cache/multispeaker_train_split.json
    { "train": [video_id, ...], "val": [video_id, ...] }
  Split policy: leave ONE video per speaker as val (plan A).

Dedup text with the same n-gram collapser used on the test set (YouTube
auto-captions on train side have the same contiguous repetition issue).

Run:
  cd /root/autodl-tmp/vsr_bench
  python prep_multispeaker_train_cache.py --limit-per-speaker 3   # smoke
  python prep_multispeaker_train_cache.py                         # full
"""
import argparse
import csv
import json
import os
import random
import sys

import cv2
import numpy as np
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE

CLIPS_ROOT = "/root/autodl-tmp/speaker_clips/train"
ROI_ROOT = "/root/autodl-tmp/speaker_roi/multi_speaker/multi_speaker_video_seg16s/train"
NAME = "multispeaker_train"
SPLIT_SEED = 0


# ---- dedup helper (same algo as fix_manifest_dedup.py) ----
def dedup_repeated_ngrams(text, max_n=15):
    words = text.split()
    out = []
    i = 0
    n = len(words)
    while i < n:
        collapsed = False
        max_try = min(max_n, (n - i) // 2)
        for k in range(max_try, 0, -1):
            if words[i:i + k] == words[i + k:i + 2 * k]:
                reps = 2
                while words[i:i + k] == words[i + reps * k:i + (reps + 1) * k]:
                    reps += 1
                out.extend(words[i:i + k])
                i += reps * k
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def collect_rows(limit_per_speaker=0):
    rows = []
    for sp in sorted(os.listdir(CLIPS_ROOT)):
        sp_dir = os.path.join(CLIPS_ROOT, sp)
        if not os.path.isdir(sp_dir):
            continue
        kept = 0
        for vid in sorted(os.listdir(sp_dir)):
            mf = os.path.join(sp_dir, vid, "manifest.csv")
            if not os.path.exists(mf):
                continue
            with open(mf, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    cid = r["clip_id"]
                    roi_mp4 = os.path.join(ROI_ROOT, sp, vid, f"{cid}.mp4")
                    if not os.path.exists(roi_mp4):
                        continue
                    rows.append({
                        "utt_id": f"{sp}__{vid}__{cid}",
                        "speaker": sp,
                        "video_id": vid,
                        "clip_id": cid,
                        "text": dedup_repeated_ngrams(r["text"].strip()),
                        "roi_path": roi_mp4,
                    })
                    kept += 1
                    if limit_per_speaker and kept >= limit_per_speaker:
                        break
            if limit_per_speaker and kept >= limit_per_speaker:
                break
        print(f"  {sp}: {kept} clips")
    return rows


def convert_one(roi_mp4, out_npy):
    v = torchvision.io.read_video(roi_mp4, pts_unit="sec")[0].numpy()
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in v]).astype(np.uint8)
    if g.shape[1:] != (96, 96):
        g = np.stack([cv2.resize(f, (96, 96)) for f in g])
    np.save(out_npy, g)
    return g.shape[0]  # frames


def make_split(rows, seed=SPLIT_SEED):
    """plan A: leave one video per speaker as val."""
    rng = random.Random(seed)
    by_sp_vids = {}
    for r in rows:
        by_sp_vids.setdefault(r["speaker"], set()).add(r["video_id"])
    train_vids, val_vids = [], []
    for sp, vids in sorted(by_sp_vids.items()):
        vids = sorted(vids)
        held = rng.choice(vids)
        val_vids.append(held)
        for v in vids:
            if v != held:
                train_vids.append(v)
    return {"train": sorted(train_vids), "val": sorted(val_vids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    a = ap.parse_args()

    out_dir = os.path.join(CACHE, NAME)
    os.makedirs(out_dir, exist_ok=True)

    print(f"collecting train rows (limit_per_speaker={a.limit_per_speaker}) ...")
    rows = collect_rows(a.limit_per_speaker)
    print(f"\ntotal {len(rows)} clips across {len({r['speaker'] for r in rows})} speakers\n")

    ok, bad = [], []
    for i, r in enumerate(rows):
        npy_name = f"{r['utt_id']}.npy"
        out_npy = os.path.join(out_dir, npy_name)
        try:
            if a.skip_existing and os.path.exists(out_npy):
                arr = np.load(out_npy, mmap_mode="r")
                frames = arr.shape[0]
            else:
                frames = convert_one(r["roi_path"], out_npy)
                if (i + 1) % 200 == 0 or i < 3:
                    print(f"[{i+1}/{len(rows)}] {r['utt_id']} frames={frames}")
            ok.append({
                "utt_id": r["utt_id"],
                "clip_id": r["clip_id"],
                "video_id": r["video_id"],
                "speaker": r["speaker"],
                "path": r["roi_path"],
                "npy": npy_name,
                "text": r["text"],
                "frames": frames,
            })
        except Exception as e:
            bad.append(r["utt_id"])
            print(f"[{i+1}/{len(rows)}] {r['utt_id']} FAILED: {e}")

    # manifest_frames.csv (what adapt_autoavsr.py reads)
    mf_frames = os.path.join(out_dir, "manifest_frames.csv")
    with open(mf_frames, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "utt_id", "clip_id", "video_id", "speaker", "path", "npy", "text", "frames"])
        w.writeheader()
        w.writerows(ok)

    # manifest.csv (compatibility w/ eval scripts)
    mf = os.path.join(out_dir, "manifest.csv")
    with open(mf, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy", "speaker", "video"])
        w.writeheader()
        for r in ok:
            w.writerow({"utt_id": r["utt_id"], "text": r["text"],
                        "npy": r["npy"], "speaker": r["speaker"],
                        "video": r["video_id"]})

    # split.json
    split = make_split(ok)
    split_path = os.path.join(CACHE, f"{NAME}_split.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    # summary
    tr_clips = sum(1 for r in ok if r["video_id"] in set(split["train"]))
    va_clips = sum(1 for r in ok if r["video_id"] in set(split["val"]))
    tr_min = sum(r["frames"] for r in ok if r["video_id"] in set(split["train"])) / 25 / 60
    va_min = sum(r["frames"] for r in ok if r["video_id"] in set(split["val"])) / 25 / 60
    print(f"\nDONE: {len(ok)} ok, {len(bad)} failed")
    print(f"manifest_frames -> {mf_frames}")
    print(f"split           -> {split_path}")
    print(f"  train: {len(split['train'])} videos, {tr_clips} clips, {tr_min:.1f} min")
    print(f"  val:   {len(split['val'])} videos, {va_clips} clips, {va_min:.1f} min")
    if bad:
        print("failed ids:", bad[:20], "..." if len(bad) > 20 else "")


if __name__ == "__main__":
    main()
