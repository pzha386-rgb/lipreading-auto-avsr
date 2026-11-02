# -*- coding: utf-8 -*-
"""
cache_tomscott_rois.py  -  mouth ROIs for the kept Tom Scott clips

Uses YOUR cache_bbc_rois.from_raw() unchanged (RetinaFace -> AWing 98 -> 68 points
-> Auto-AVSR VideoProcess -> 96x96 gray), so the data is made exactly like the BBC test set.

Added here:
- unique ids: <video_id>__<clip_id>   (every video has a clip_0000)
- N parallel processes on the GPU, live progress, merged manifest at the end
- manifest has an extra video_id column (needed for the train / test split)

Put this file in /root/autodl-tmp/vsr_bench/ and run from there:
  python cache_tomscott_rois.py --limit 12 --shards 2      # quick test (12 clips)
  nohup python cache_tomscott_rois.py --shards 6 > /root/autodl-tmp/logs/roi_tomscott.log 2>&1 &
Output: /root/autodl-tmp/cache/tomscott/<id>.npy + manifest.csv
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from argparse import Namespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--keep", default="/root/autodl-tmp/tomscott/clips/keep_manifest.csv")
ap.add_argument("--clips-dir", default="/root/autodl-tmp/tomscott/clips")
ap.add_argument("--name", default="tomscott")
ap.add_argument("--shards", type=int, default=6)
ap.add_argument("--shard", type=int, default=-1, help="internal: which shard this process runs")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

from common import CACHE   # noqa: E402

OUT = os.path.join(CACHE, a.name)
LOGD = "/root/autodl-tmp/logs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGD, exist_ok=True)


def load_rows():
    with open(a.keep, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[:a.limit]
    return [{"clip_id": f"{r['video_id']}__{r['clip_id']}", "path": r["path"],
             "text": r["text"], "video_id": r["video_id"]} for r in rows]


def run_shard(i):
    from cache_bbc_rois import from_raw
    rows = load_rows()[i::a.shards]
    vid_of = {r["clip_id"]: r["video_id"] for r in rows}
    ok, bad = from_raw(Namespace(clips_dir=a.clips_dir), rows, OUT)
    with open(os.path.join(OUT, f"_shard{i}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy", "video_id"])
        w.writeheader()
        for r in ok:
            w.writerow({**r, "video_id": vid_of[r["utt_id"]]})
    with open(os.path.join(OUT, f"_shard{i}_failed.txt"), "w") as f:
        f.write("\n".join(bad))


def launch():
    rows = load_rows()
    total = len(rows)
    print(f"{total} clips -> {OUT}   ({a.shards} parallel processes)")
    procs = []
    for i in range(a.shards):
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--shard", str(i), "--shards", str(a.shards),
               "--keep", a.keep, "--clips-dir", a.clips_dir, "--name", a.name]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        log = open(os.path.join(LOGD, f"roi_{a.name}_shard{i}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE))
        time.sleep(3)   # do not load all models at the same second

    t0 = time.time()
    while any(p.poll() is None for p in procs):
        done = len(glob.glob(os.path.join(OUT, "*.npy")))
        el = time.time() - t0
        eta = el / max(done, 1) * (total - done) if done else 0
        print(f"\r  {done}/{total} clips  ({100 * done / max(total, 1):.0f}%)   "
              f"elapsed {el / 60:.0f} min   left ~{eta / 60:.0f} min   ", end="", flush=True)
        time.sleep(20)
    print()

    merged, failed = [], []
    for i in range(a.shards):
        p = os.path.join(OUT, f"_shard{i}.csv")
        if os.path.exists(p):
            merged += list(csv.DictReader(open(p, encoding="utf-8")))
        else:
            print(f"shard {i} has no output - see {LOGD}/roi_{a.name}_shard{i}.log")
        fp = os.path.join(OUT, f"_shard{i}_failed.txt")
        if os.path.exists(fp):
            failed += [l for l in open(fp).read().split("\n") if l]
    merged.sort(key=lambda r: r["utt_id"])
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy", "video_id"])
        w.writeheader()
        w.writerows(merged)
    print(f"DONE: {len(merged)} ok, {len(failed)} failed  ->  {OUT}/manifest.csv")
    print(f"total time {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    if a.shard >= 0:
        run_shard(a.shard)
    else:
        launch()
