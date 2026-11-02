# -*- coding: utf-8 -*-
"""
eval_autoavsr_nbest.py  -  方法 A：beam search 置信度（T1 / Auto-AVSR）

放在 /root/autodl-tmp/vsr_bench/ 里（和 eval_autoavsr_cache.py 同一个文件夹）
- 模型加载：直接复用你的 eval_autoavsr_cache.load_model()（同样的改名逻辑）
- 数据：同样的 ROI 缓存 + VideoTransform(subset="test")
- 解码：和 ModelModule.forward() 同一条路径，额外保留前 5 个结果和分数
- 输出：
    /root/autodl-tmp/day1/conf/T1_<dataset>_confA.csv   (给 conf_analysis.py 用)
    /root/autodl-tmp/day1/conf/T1_<dataset>_nbest.jsonl (第 3 天 LLM 纠错用)

运行：
  python -u eval_autoavsr_nbest.py --ckpt <T1 权重路径> --dataset bbc --limit 5
  python -u eval_autoavsr_nbest.py --ckpt <T1 权重路径> --dataset bbc
  python -u eval_autoavsr_nbest.py --ckpt <T1 权重路径> --dataset lrs3_300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/root/autodl-tmp/day1")
from common import read_manifest, score                 # 你的 vsr_bench/common.py
from eval_autoavsr_cache import load_model              # 你的加载函数（会把 auto_avsr 加入路径）
from conf_utils import decode_nbest_autoavsr, nbest_features

ap = argparse.ArgumentParser()
ap.add_argument("--id", default="T1")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--convert-fn", default="")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--outdir", default="/root/autodl-tmp/day1/conf")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"===== {a.id} N-best on {a.dataset} =====")
model = load_model(a, device)
from datamodule.transforms import VideoTransform
vt = VideoTransform(subset="test")

items = read_manifest(a.dataset)
if a.limit:
    items = items[:a.limit]

tag = f"{a.id}_{a.dataset}" + (f"_limit{a.limit}" if a.limit else "")
rows, tot_e, tot_n, t_all = [], 0, 0, time.time()
fj = open(os.path.join(a.outdir, f"{tag}_nbest.jsonl"), "w", encoding="utf-8")

for i, (uid, ref, npy) in enumerate(items):
    nbest, sec = [], 0.0
    try:
        vid = vt(torch.from_numpy(np.load(npy)).unsqueeze(1).float()).to(device)
        t0 = time.perf_counter()
        nbest = decode_nbest_autoavsr(model, vid, k=a.k)
        if device == "cuda":
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
    except Exception as e:
        print(f"[{i}] {uid} ERROR: {e}")

    hyp = nbest[0]["text"] if nbest else ""
    rn, hn, err, n = score(ref, hyp)
    tot_e, tot_n = tot_e + err, tot_n + n
    feats = nbest_features(nbest)
    rows.append({"id": uid, "ref": rn, "hyp": hn, "errors": err, "n_words": n,
                 "seconds": round(sec, 3), **feats})
    fj.write(json.dumps({"id": uid, "ref": ref, "nbest": nbest, **feats}, ensure_ascii=False) + "\n")

    if a.limit or (i + 1) % 20 == 0:
        print(f"[{i + 1}/{len(items)}] WER so far {100 * tot_e / max(tot_n, 1):.2f}%  ({sec:.2f}s)  "
              f"logp={feats['confA_logp']:.3f} margin={feats['confA_margin']:.3f} nbest={feats['confA_nbest']:.2f}")
        if a.limit:
            print(f"   REF: {rn}\n   HYP: {hn}")
            try:
                with torch.no_grad():
                    orig = str(model(vid)).strip()
                print(f"   原来的 model(vid) 输出: {'一致 OK' if orig == hyp else '不一致! ' + orig}")
            except Exception as e:
                print(f"   原来的 model(vid) 检查失败: {e}")
            for j, h in enumerate(nbest[1:3], 2):
                print(f"   #{j}: {h['text']}   (score {h['score']:.2f})")
fj.close()

out = os.path.join(a.outdir, f"{tag}_confA.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nfinal WER {100 * tot_e / max(tot_n, 1):.2f}%   total {time.time() - t_all:.0f}s")
print(f"saved {out}")
print("检查：完整运行时，这个 WER 应该和 table_main 里 T1 一样（bbc 25.47 / lrs3_300 25.16）")
