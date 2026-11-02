# -*- coding: utf-8 -*-
"""
Auto-AVSR 统一评测（S0、S1、S6、T1；读 ROI 缓存）

推理路径和你之前的一样：ModelModule.forward()，VideoTransform(subset="test")
权重键名：HuggingFace 版用 espnet 原生键名（encoder.frontend.* 等），
         这里先直接加载，不行再自动改名；改名后还缺键就停止并报告，不会"带病"跑。

运行：
  python eval_autoavsr_cache.py --id S0 --ckpt /root/autodl-tmp/checkpoints/vsr_trlrs3_base.pth --dataset bbc --limit 5
  # 如果你找到了以前写的 convert_ckpt 函数，可以直接用它：
  python eval_autoavsr_cache.py ... --convert-fn 你的文件名:convert_ckpt
"""
import argparse
import importlib
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch

from common import CODE_AUTOAVSR, RESULTS, read_manifest, save_results, score

sys.path.insert(0, CODE_AUTOAVSR)


def remap(sd, model_keys, model_shapes):
    """把 espnet 原生键名改成 GitHub 代码的键名"""
    out = {}
    for k, v in sd.items():
        k2 = k
        for a, b in [("encoder.frontend.", "frontend."), ("module.", ""), ("model.", "")]:
            if k2.startswith(a) and k2 not in model_keys:
                k2 = b + k2[len(a):]
        out[k2] = v
    # 还对不上的：按"后缀 + 形状"唯一匹配（例如 proj_encoder）
    missing = [k for k in model_keys if k not in out]
    extra = {k: v for k, v in out.items() if k not in model_keys}
    for mk in missing:
        leaf = ".".join(mk.split(".")[-2:])
        c = [ek for ek, ev in extra.items() if ek.endswith(leaf) and tuple(ev.shape) == model_shapes[mk]]
        if len(c) == 1:
            out[mk] = extra.pop(c[0])
            out.pop(c[0], None)
    # 第二轮：剩下的按"形状"唯一匹配（例如 proj_encoder 在 espnet 版里叫别的名字）
    missing = [k for k in model_keys if k not in out]
    extra = {k: v for k, v in out.items() if k not in model_keys}
    for mk in missing:
        c = [ek for ek, ev in extra.items() if tuple(ev.shape) == model_shapes[mk]]
        if len(c) == 1:
            print(f"  按形状匹配: {c[0]}  ->  {mk}  {model_shapes[mk]}")
            out[mk] = extra.pop(c[0])
            out.pop(c[0], None)
        else:
            print(f"  无法唯一匹配 {mk} {model_shapes[mk]}，候选: {c[:5]}")
    return out


def load_model(a, device):
    from lightning import ModelModule
    mm = ModelModule(Namespace(modality="video", pretrained_model_path=None, ctc_weight=0.1))
    sd = torch.load(a.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if a.convert_fn:
        mod, fn = a.convert_fn.split(":")
        sd = getattr(importlib.import_module(mod), fn)(sd)
    model_sd = mm.model.state_dict()
    try:
        mm.model.load_state_dict(sd, strict=True)
        print("权重直接加载成功")
    except RuntimeError:
        sd = remap(sd, set(model_sd), {k: tuple(v.shape) for k, v in model_sd.items()})
        res = mm.model.load_state_dict(sd, strict=False)
        if res.missing_keys:
            print("缺少的键（前 10 个）:", res.missing_keys[:10])
            raise SystemExit(f"权重键名对不上（缺 {len(res.missing_keys)} 个）。"
                             "请用 --convert-fn 指定你以前的 convert_ckpt 函数，或把这段输出发给 Claude。")
        print(f"权重改名后加载成功（多余键 {len(res.unexpected_keys)} 个已忽略）")
    return mm.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--convert-fn", default="", help="可选：模块名:函数名")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"===== {a.id} on {a.dataset} =====")
    model = load_model(a, device)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"参数量: {n_params / 1e6:.1f} M")
    from datamodule.transforms import VideoTransform
    vt = VideoTransform(subset="test")

    items = read_manifest(a.dataset)
    if a.limit:
        items = items[:a.limit]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, t_all = [], time.time()
    for i, (uid, ref, npy) in enumerate(items):
        hyp, sec = "", 0.0
        try:
            vid = vt(torch.from_numpy(np.load(npy)).unsqueeze(1).float()).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                hyp = str(model(vid)).strip()
            if device == "cuda":
                torch.cuda.synchronize()
            sec = time.perf_counter() - t0
        except Exception as e:
            print(f"[{i}] {uid} ERROR: {e}")
        rn, hn, err, n = score(ref, hyp)
        rows.append({"utt_id": uid, "ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": hn,
                     "errors": err, "n_words": n, "seconds": round(sec, 3)})
        if a.limit or (i + 1) % 20 == 0:
            tot_e = sum(r["errors"] for r in rows)
            tot_n = sum(r["n_words"] for r in rows)
            print(f"[{i + 1}/{len(items)}] WER so far {100 * tot_e / max(tot_n, 1):.2f}%  ({sec:.2f}s)")
            if a.limit:
                print(f"   REF: {rn}\n   HYP: {hn}")

    out = a.out or os.path.join(RESULTS, f"{a.id}_{a.dataset}.csv")
    if a.limit:
        out = out.replace(".csv", f"_limit{a.limit}.csv")
    save_results(out, rows, {
        "model_id": a.id, "dataset": a.dataset, "ckpt": a.ckpt, "params_M": round(n_params / 1e6, 1),
        "peak_gpu_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
        "total_seconds": round(time.time() - t_all, 1)})


if __name__ == "__main__":
    main()
