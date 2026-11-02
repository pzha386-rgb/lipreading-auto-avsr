# -*- coding: utf-8 -*-
"""
导出 GUI 需要的所有数据 -> /root/autodl-tmp/gui_data.zip
  results/   BBC 和 LRS3(300) 的逐句识别结果
  videos/bbc/        BBC 原始视频（人脸）
  videos/lrs3_300/   LRS3 的嘴部 ROI 视频（从缓存生成，放大到 288x288）
  figures/   论文图
  models.csv

运行：
  pip install -q "imageio[ffmpeg]"
  python export_gui.py
"""
import csv
import glob
import os
import shutil
import zipfile

import numpy as np

from common import CACHE, HERE, RESULTS, ROOT

OUT = f"{ROOT}/gui_data"
shutil.rmtree(OUT, ignore_errors=True)
for d in ["results", "videos/bbc", "videos/lrs3_300", "figures"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

# 1) 结果（只要 GUI 用的两个数据集）
n = 0
for f in glob.glob(f"{RESULTS}/*_bbc.csv") + glob.glob(f"{RESULTS}/*_lrs3_300.csv"):
    shutil.copy(f, f"{OUT}/results/")
    n += 1
print(f"结果文件: {n}")

# 2) BBC 原始视频
bbc = sorted(glob.glob(f"{ROOT}/bbc_clips/*.mp4"))
for f in bbc:
    shutil.copy(f, f"{OUT}/videos/bbc/")
print(f"BBC 视频: {len(bbc)}")

# 3) LRS3 300 段：嘴部 ROI -> mp4（H.264，浏览器能播放）
import imageio.v2 as imageio
import cv2

rows = list(csv.DictReader(open(f"{CACHE}/lrs3_300/manifest.csv", encoding="utf-8")))
for r in rows:
    arr = np.load(os.path.join(CACHE, "lrs3_300", r["npy"]))
    frames = [cv2.cvtColor(cv2.resize(f, (288, 288), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
              for f in arr]
    imageio.mimwrite(f"{OUT}/videos/lrs3_300/{r['utt_id']}.mp4", frames, fps=25,
                     codec="libx264", quality=7, macro_block_size=16)
print(f"LRS3 ROI 视频: {len(rows)}")

# 4) 图和模型清单
for f in glob.glob(f"{ROOT}/figures/*.png"):
    shutil.copy(f, f"{OUT}/figures/")
for f in glob.glob(f"{ROOT}/figures/*.csv"):
    shutil.copy(f, f"{OUT}/figures/")
shutil.copy(f"{HERE}/models.csv", OUT)

# 5) 打包
z = f"{ROOT}/gui_data.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in glob.glob(f"{OUT}/**/*", recursive=True):
        if os.path.isfile(p):
            zf.write(p, os.path.relpath(p, ROOT))
print(f"\n已打包: {z}  ({os.path.getsize(z) / 1e6:.0f} MB)")
