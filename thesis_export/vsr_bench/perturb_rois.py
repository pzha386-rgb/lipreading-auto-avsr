# -*- coding: utf-8 -*-
"""
鲁棒性测试：对 BBC 的嘴部 ROI 缓存生成扰动版本（模拟机器人摄像头会遇到的情况）

每种条件生成一个新的"数据集"：cache/bbc_{条件}/ ，可以直接用 run_queue.py 测试
  例如：python run_queue.py --datasets bbc_blur2 bbc_dark03 --ids T1 T3

条件（固定随机种子，所有模型看到完全一样的扰动）：
  blur2 / blur4     高斯模糊 sigma=2 / 4         -> 对焦不准、运动模糊
  low50 / low25     分辨率降到 50% / 25% 再放大   -> 人离摄像头远
  dark05 / dark03   亮度降到 0.5 / 0.3 倍         -> 室内光线暗
  drop25 / drop50   随机 25% / 50% 的帧被冻结     -> 网络卡顿或算力不足（用上一帧代替，长度不变）

运行：
  python perturb_rois.py                 # 生成全部 8 种
  python perturb_rois.py --only blur2    # 只生成一种
  python perturb_rois.py --preview       # 另存一张对比图 robustness_preview.png
"""
import argparse
import csv
import os

import cv2
import numpy as np

from common import CACHE

CONDITIONS = {
    "blur2": ("blur", 2.0), "blur4": ("blur", 4.0),
    "low50": ("low", 0.50), "low25": ("low", 0.25),
    "dark05": ("dark", 0.5), "dark03": ("dark", 0.3),
    "drop25": ("drop", 0.25), "drop50": ("drop", 0.50),
}


def apply(arr, kind, level, rng):
    """arr: T x 96 x 96 uint8"""
    if kind == "blur":
        return np.stack([cv2.GaussianBlur(f, (0, 0), level) for f in arr])
    if kind == "low":
        h, w = arr.shape[1:]
        s = (max(1, int(w * level)), max(1, int(h * level)))
        return np.stack([cv2.resize(cv2.resize(f, s, interpolation=cv2.INTER_AREA), (w, h),
                                    interpolation=cv2.INTER_LINEAR) for f in arr])
    if kind == "dark":
        return np.clip(arr.astype(np.float32) * level, 0, 255).astype(np.uint8)
    if kind == "drop":
        out = arr.copy()
        drop = rng.random(len(arr)) < level
        drop[0] = False                      # 第一帧保留
        for t in range(1, len(arr)):
            if drop[t]:
                out[t] = out[t - 1]          # 冻结：重复上一帧
        return out
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bbc")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()

    src_dir = os.path.join(CACHE, a.src)
    with open(os.path.join(src_dir, "manifest.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    conds = a.only or list(CONDITIONS)

    for name in conds:
        kind, level = CONDITIONS[name]
        out_dir = os.path.join(CACHE, f"{a.src}_{name}")
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        for r in rows:
            p = os.path.join(out_dir, r["npy"])
            if not os.path.exists(p):
                np.save(p, apply(np.load(os.path.join(src_dir, r["npy"])), kind, level, rng))
        with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy"])
            w.writeheader()
            w.writerows(rows)
        print(f"{a.src}_{name:7s} 完成 {len(rows)} 段")

    if a.preview:
        r = rows[0]
        base = np.load(os.path.join(src_dir, r["npy"]))
        t = len(base) // 2
        tiles = [base[t]] + [np.load(os.path.join(CACHE, f"{a.src}_{n}", r["npy"]))[t] for n in conds]
        labels = ["original"] + conds
        tiles = [cv2.putText(cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), l, (3, 12),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1) for x, l in zip(tiles, labels)]
        out = os.path.join(os.path.dirname(CACHE), "robustness_preview.png")
        cv2.imwrite(out, cv2.resize(np.hstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        print("对比图:", out, "（drop 类在单帧上看不出来，正常）")


if __name__ == "__main__":
    main()
