# -*- coding: utf-8 -*-
"""
make_split.py  -  split Tom Scott clips BY VIDEO into train / val / test

- a video is in ONE set only (no clip of a test video is ever used in training)
- test : random videos until >= --test-min minutes   (held-out new-speaker test set)
- val  : next videos until  >= --val-min minutes     (early stopping)
- train: all other videos                            (adaptation pool)
- the BBC test video was never downloaded, so it stays a second, fully separate test set

Outputs (in /root/autodl-tmp/cache/):
  tomscott_split.json
  tomscott_test/manifest.csv, tomscott_val/manifest.csv   (read_manifest("tomscott_test") works)
  tomscott/manifest_frames.csv  (manifest + number of frames)

Run in /root/autodl-tmp/vsr_bench:
  python make_split.py
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="tomscott")
ap.add_argument("--test-min", type=float, default=30)
ap.add_argument("--val-min", type=float, default=6)
ap.add_argument("--min-clips", type=int, default=10, help="videos with fewer kept clips go to train")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

src = os.path.join(CACHE, a.name)
man = pd.read_csv(os.path.join(src, "manifest.csv"), keep_default_na=False)
man["frames"] = [np.load(os.path.join(src, f), mmap_mode="r").shape[0] for f in man["npy"]]
man.to_csv(os.path.join(src, "manifest_frames.csv"), index=False)

per = man.groupby("video_id").agg(clips=("utt_id", "size"), frames=("frames", "sum"))
per["min"] = per["frames"] / 25 / 60
vids = sorted(per.index)
random.Random(a.seed).shuffle(vids)

test, val, train = [], [], []
t_min = v_min = 0.0
for v in vids:
    big = per.loc[v, "clips"] >= a.min_clips
    if big and t_min < a.test_min:
        test.append(v)
        t_min += per.loc[v, "min"]
    elif big and v_min < a.val_min:
        val.append(v)
        v_min += per.loc[v, "min"]
    else:
        train.append(v)
tr_min = per.loc[train, "min"].sum()

json.dump({"test": sorted(test), "val": sorted(val), "train": sorted(train), "seed": a.seed},
          open(os.path.join(CACHE, f"{a.name}_split.json"), "w"), indent=2)

for part, vs in [("test", test), ("val", val)]:
    d = os.path.join(CACHE, f"{a.name}_{part}")
    os.makedirs(d, exist_ok=True)
    sub = man[man["video_id"].isin(vs)].copy()
    sub["npy"] = [f"../{a.name}/{f}" for f in sub["npy"]]       # point to the shared npy files
    sub[["utt_id", "text", "npy", "video_id"]].to_csv(os.path.join(d, "manifest.csv"), index=False)

print(f"TEST : {len(test):3d} videos  {(man['video_id'].isin(test)).sum():5d} clips  {t_min:6.1f} min")
print(f"VAL  : {len(val):3d} videos  {(man['video_id'].isin(val)).sum():5d} clips  {v_min:6.1f} min")
print(f"TRAIN: {len(train):3d} videos  {(man['video_id'].isin(train)).sum():5d} clips  {tr_min:6.1f} min")
print(f"saved {CACHE}/{a.name}_split.json, {a.name}_test/, {a.name}_val/")
