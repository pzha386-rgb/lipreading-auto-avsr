# -*- coding: utf-8 -*-
"""
把 HuggingFace 的 LRS3 parquet 转成统一的 ROI 缓存（只需运行一次）
输出：/root/autodl-tmp/cache/lrs3/{utt_id}.npy  （T x 96 x 96, uint8 灰度）
      /root/autodl-tmp/cache/lrs3/manifest.csv （utt_id, text, npy）

运行：
  python prep_lrs3_cache.py --parquet-dir /root/autodl-tmp/data/lrs3_test_hf/data
  python prep_lrs3_cache.py --subset 300      # 另外生成固定的 300 段子集 lrs3_300
"""
import argparse
import csv
import glob
import os
import random

import numpy as np
import pandas as pd

from common import CACHE


def frames_to_array(video_frames):
    """和你之前的 prep_video 一样：嵌套的对象数组 -> T x H x W uint8"""
    frames = []
    for f in video_frames:
        rows = [np.asarray(r, dtype=np.uint8) for r in f]
        frames.append(np.stack(rows))
    arr = np.stack(frames)
    if arr.ndim == 4:                 # 万一是 T x H x W x 3
        arr = arr.mean(axis=-1).astype(np.uint8)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="/root/autodl-tmp/data/lrs3_test_hf/data")
    ap.add_argument("--subset", type=int, default=300, help="同时生成固定随机子集的大小，0 = 不生成")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    out_dir = os.path.join(CACHE, "lrs3")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.parquet_dir, "*.parquet")))
    print("parquet 文件:", files)

    rows, k = [], 0
    for fp in files:
        df = pd.read_parquet(fp)
        text_col = next((c for c in ["label", "text", "transcript", "sentence"] if c in df.columns), None)
        id_col = next((c for c in ["id", "key", "name", "utt_id"] if c in df.columns), None)
        if text_col is None or "video" not in df.columns:
            raise SystemExit(f"列名不对: {df.columns.tolist()}，请把这行输出发给 Claude")
        for _, r in df.iterrows():
            uid = str(r[id_col]).replace("/", "_") if id_col else f"lrs3_{k:04d}"
            npy = f"{uid}.npy"
            if not os.path.exists(os.path.join(out_dir, npy)):
                np.save(os.path.join(out_dir, npy), frames_to_array(r["video"]))
            rows.append({"utt_id": uid, "text": str(r[text_col]).strip(), "npy": npy})
            k += 1
            if k % 200 == 0:
                print(f"  {k} done")

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy"])
        w.writeheader()
        w.writerows(rows)
    print(f"LRS3 缓存完成: {len(rows)} 段 -> {out_dir}")
    sample = np.load(os.path.join(out_dir, rows[0]["npy"]))
    print("第一段形状:", sample.shape, sample.dtype, "（应该是 T x 96 x 96 uint8）")

    # 固定子集：所有模型都用同一批 300 段，结果可以互相比较
    if a.subset > 0:
        sub_dir = os.path.join(CACHE, f"lrs3_{a.subset}")
        os.makedirs(sub_dir, exist_ok=True)
        sub = sorted(random.Random(a.seed).sample(rows, min(a.subset, len(rows))), key=lambda r: r["utt_id"])
        with open(os.path.join(sub_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy"])
            w.writeheader()
            for r in sub:
                w.writerow({**r, "npy": os.path.join("..", "lrs3", r["npy"])})
        print(f"子集完成: {len(sub)} 段 -> {sub_dir}（seed={a.seed}）")


if __name__ == "__main__":
    main()
