# -*- coding: utf-8 -*-
"""
VSR-ML 仓库的模型评测（S2 / S4 / S5，及加语言模型的 S2lm / S4lm / S5lm；读 ROI 缓存）

和官方 eval.py 一样：pipelines.model.AVSR + VideoTransform，解码参数从官方 .ini 读取
不同之处：跳过人脸检测，直接读统一的 ROI 缓存（官方流程也是 RetinaFace + 灰度 96x96，格式相同）

运行：
  python eval_vsrml.py --id S4 --model-dir /root/autodl-tmp/checkpoints/new/S4_vsrml_lrs3 \
      --ini LRS3_V_WER32.3 --dataset bbc --limit 5
  # 加语言模型
  python eval_vsrml.py --id S4lm ... --lm-dir /root/autodl-tmp/checkpoints/new/LM_vsrml
"""
import argparse
import glob
import os
import sys
import time
from configparser import ConfigParser

import numpy as np
import torch

from common import RESULTS, ROOT, read_manifest, save_results, score

CODE_VSRML = os.environ.get("CODE_VSRML", f"{ROOT}/code/vsr_ml")
sys.path.insert(0, CODE_VSRML)


def find_pair(d, what):
    """在文件夹里找 model.pth + model.json（zip 解压后可能多一层文件夹）"""
    pths = [p for p in glob.glob(os.path.join(d, "**", "*.pth"), recursive=True)]
    for p in pths:
        j = os.path.join(os.path.dirname(p), "model.json")
        if os.path.exists(j):
            return p, j
    raise SystemExit(f"在 {d} 里找不到 {what} 的 .pth + model.json。里面有: "
                     f"{glob.glob(os.path.join(d, '**', '*'), recursive=True)[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model-dir", required=True, help="解压后的模型文件夹")
    ap.add_argument("--ini", required=True, help="官方配置名，例如 LRS3_V_WER32.3")
    ap.add_argument("--lm-dir", default="", help="语言模型文件夹（不填 = 不用 LM）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ini_path = os.path.join(CODE_VSRML, "configs", f"{a.ini}.ini")
    cfg = ConfigParser()
    cfg.read(ini_path)
    beam = cfg.getint("decode", "beam_size")
    ctc = cfg.getfloat("decode", "ctc_weight")
    penalty = cfg.getfloat("decode", "penalty")
    lm_weight = cfg.getfloat("decode", "lm_weight") if a.lm_dir else 0.0
    print(f"===== {a.id} on {a.dataset} =====")
    print(f"官方配置 {a.ini}: beam={beam} ctc={ctc} penalty={penalty} "
          f"lm_weight(ini)={cfg.get('decode', 'lm_weight')} -> 本次 lm_weight={lm_weight}")

    model_path, model_conf = find_pair(a.model_dir, "模型")
    rnnlm, rnnlm_conf = find_pair(a.lm_dir, "语言模型") if a.lm_dir else (None, None)
    print("模型:", model_path)
    print("语言模型:", rnnlm)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from pipelines.data.transforms import VideoTransform
    from pipelines.model import AVSR
    avsr = AVSR("video", model_path, model_conf, rnnlm, rnnlm_conf, penalty, ctc, lm_weight, beam, device)
    vt = VideoTransform(speed_rate=1.0)
    n_params = sum(p.numel() for p in avsr.model.parameters())
    print(f"参数量: {n_params / 1e6:.1f} M")

    items = read_manifest(a.dataset)
    if a.limit:
        items = items[:a.limit]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    rows, t_all = [], time.time()
    for i, (uid, ref, npy) in enumerate(items):
        hyp, sec = "", 0.0
        try:
            data = vt(torch.tensor(np.load(npy)))      # 和官方 load_data 一样：T x 96 x 96 灰度 -> transform
            t0 = time.perf_counter()
            hyp = avsr.infer(data)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            sec = time.perf_counter() - t0
        except Exception as e:
            print(f"[{i}] {uid} ERROR: {e}")
        rn, hn, err, n = score(ref, hyp)
        rows.append({"utt_id": uid, "ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": hn,
                     "errors": err, "n_words": n, "seconds": round(sec, 3)})
        if a.limit or (i + 1) % 20 == 0:
            te, tn = sum(r["errors"] for r in rows), sum(r["n_words"] for r in rows)
            print(f"[{i + 1}/{len(items)}] WER so far {100 * te / max(tn, 1):.2f}%  ({sec:.2f}s)")
            if a.limit:
                print(f"   REF: {rn}\n   HYP: {hn}")

    out = a.out or os.path.join(RESULTS, f"{a.id}_{a.dataset}.csv")
    if a.limit:
        out = out.replace(".csv", f"_limit{a.limit}.csv")
    save_results(out, rows, {
        "model_id": a.id, "dataset": a.dataset, "model_path": model_path, "ini": a.ini,
        "lm": rnnlm, "lm_weight": lm_weight, "beam": beam, "params_M": round(n_params / 1e6, 1),
        "peak_gpu_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else None,
        "total_seconds": round(time.time() - t_all, 1)})


if __name__ == "__main__":
    main()
