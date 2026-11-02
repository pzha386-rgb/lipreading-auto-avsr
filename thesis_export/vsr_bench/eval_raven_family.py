# -*- coding: utf-8 -*-
"""
RAVEn / BRAVEn 统一评测（替代 eval_braven_lrs3.py，读 ROI 缓存）

新功能：
  1. 自动从权重里读出模型结构（低资源/高资源的解码器大小不同，不用手动改配置）
  2. --lm 语言模型（R1、R3、B2、B5 用）
  3. 记录每段耗时、显存峰值、参数量（速度对比用）
  4. 输出统一格式，GUI 和汇总脚本直接读

做法和你之前跑通的一样：编码器在 GPU，beam search 在 CPU（BatchBeamSearch 搬不到 GPU）

运行：
  python eval_raven_family.py --id T3 --ckpt <权重> --dataset bbc --limit 5
  python eval_raven_family.py --id B2 --ckpt <权重> --lm <LM权重> --dataset lrs3
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch

from common import CODE_RAVEN, RESULTS, read_manifest, save_results, score

sys.path.insert(0, CODE_RAVEN)

CROP, MEAN, STD = 88, 0.421, 0.165     # 和你之前的 prep_video 一样


def prep(arr):
    """T x 96 x 96 uint8 -> [1, T, 88, 88]"""
    arr = arr.astype(np.float32) / 255.0
    T, H, W = arr.shape
    i, j = (H - CROP) // 2, (W - CROP) // 2
    arr = (arr[:, i:i + CROP, j:j + CROP] - MEAN) / STD
    return torch.from_numpy(arr).unsqueeze(0)


def max_layer(sd, prefix):
    idx = [int(m.group(1)) for k in sd for m in [re.match(rf"{re.escape(prefix)}(\d+)\.", k)] if m]
    return max(idx) + 1 if idx else None


def infer_arch(ckpt_path):
    """从权重形状读出编码器/解码器的层数和维度"""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    arch = {}
    q = "encoder.encoders.0.self_attn.linear_q.weight"
    if q in sd:
        arch["adim"] = sd[q].shape[0]
        arch["eunits"] = sd["encoder.encoders.0.feed_forward.w_1.weight"].shape[0]
        arch["elayers"] = max_layer(sd, "encoder.encoders.")
    q = "decoder.decoders.0.self_attn.linear_q.weight"
    if q in sd:
        arch["ddim"] = sd[q].shape[0]
        arch["dunits"] = sd["decoder.decoders.0.feed_forward.w_1.weight"].shape[0]
        arch["dlayers"] = max_layer(sd, "decoder.decoders.")
    return arch


def find_lm_weight():
    """从 RAVEn 仓库里 *_lm*.sh 读出官方用的 lm_weight"""
    ws = set()
    for f in glob.glob(f"{CODE_RAVEN}/scripts/**/*lm*.sh", recursive=True):
        ws |= set(re.findall(r"lm_weight=([0-9.]+)", open(f, encoding="utf-8", errors="ignore").read()))
    return sorted(ws)


def build(a):
    from omegaconf import OmegaConf
    base = f"{CODE_RAVEN}/conf"
    cfg = OmegaConf.create({
        "data": {"modality": "video", "labels_type": "unigram1000"},
        "model": OmegaConf.merge(
            OmegaConf.load(f"{base}/model/finetune.yaml"),
            {"visual_backbone": OmegaConf.load(f"{base}/model/visual_backbone/resnet_transformer_{a.size}.yaml"),
             "language_model": OmegaConf.load(f"{base}/model/language_model/default.yaml"),
             "pretrained_model_path": a.ckpt,
             "pretrained_lm_path": a.lm or None}),
        "decode": OmegaConf.load(f"{base}/decode/default.yaml"),
    })
    vb = cfg.model.visual_backbone
    # 默认：解码器和编码器同尺寸（你之前跑通的低资源设置）
    vb.ddim, vb.dheads, vb.dunits = vb.adim, vb.aheads, vb.eunits
    # 用权重里的真实结构覆盖（高资源版本的解码器更大）
    arch = infer_arch(a.ckpt)
    for k, v in arch.items():
        if v is not None:
            vb[k] = v
    if "ddim" in arch and arch["ddim"] != vb.adim:
        vb.dheads = max(1, arch["ddim"] // 64)
    print("模型结构（从权重读出）:", arch)

    cfg.decode.beam_size = a.beam
    if a.lm:
        if a.lm_weight is None:
            found = find_lm_weight()
            print("RAVEn 仓库脚本里的 lm_weight:", found or "没找到")
            a.lm_weight = float(found[0]) if len(found) == 1 else 0.1
        cfg.decode.lm_weight = a.lm_weight
        print(f"使用语言模型，lm_weight = {a.lm_weight}")
    else:
        cfg.decode.lm_weight = 0.0

    from finetune_learner import Learner
    learner = Learner(cfg).to("cpu").eval()
    return cfg, learner, arch


def to_text(nbest, token_list):
    try:
        from espnet.asr.asr_utils import add_results_to_json
        s = add_results_to_json([h.asdict() for h in nbest[:1]], token_list)
    except Exception:
        y = nbest[0].yseq.tolist()[1:-1]
        s = "".join(token_list[int(i)] for i in y)
    return s.replace("▁", " ").replace("<space>", " ").replace("<eos>", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="models.csv 里的编号，例如 B2")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, help="lrs3 / lrs3_300 / bbc / self / bbc_blur2 ...")
    ap.add_argument("--lm", default="", help="语言模型权重路径（不填 = 不用 LM）")
    ap.add_argument("--lm-weight", type=float, default=None)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--size", default="large", help="base / large（读基础配置用）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--gpu-decode", action="store_true", help="试验：beam search 也放 GPU")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"===== {a.id} on {a.dataset} =====")
    cfg, learner, arch = build(a)
    dec_dev = "cpu"
    if a.gpu_decode and device == "cuda":
        # 像 USR 2.0 / VSR-ML 那样：先把整个模型搬到 GPU，再重新创建 beam search
        learner = learner.to(device)
        learner.beam_search = learner.get_beam_search(learner.model)
        learner.beam_search.to(device)
        dec_dev = device
        print("beam search 在 GPU 上")
    else:
        learner.model.encoder = learner.model.encoder.to(device)
    n_params = sum(p.numel() for p in learner.model.parameters())
    print(f"参数量: {n_params / 1e6:.1f} M")

    from utils import UNIGRAM1000_LIST
    items = read_manifest(a.dataset)
    if a.limit:
        items = items[:a.limit]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, t_all = [], time.time()
    for i, (uid, ref, npy) in enumerate(items):
        hyp, sec = "", 0.0
        try:
            v = prep(np.load(npy)).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                feat, _ = learner.model.encoder(v, None)
                nbest = learner.beam_search(x=feat.squeeze(0).to(dec_dev),
                                            maxlenratio=cfg.decode.maxlenratio,
                                            minlenratio=cfg.decode.minlenratio)
            hyp = to_text(nbest, UNIGRAM1000_LIST)
            sec = time.perf_counter() - t0
        except Exception as e:
            print(f"[{i}] {uid} ERROR: {e}")
            if i == 0:
                import traceback
                traceback.print_exc()
        rn, hn, err, n = score(ref, hyp)
        rows.append({"utt_id": uid, "ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": hn,
                     "errors": err, "n_words": n, "seconds": round(sec, 3)})
        if a.limit or (i + 1) % 20 == 0:
            tot_e = sum(r["errors"] for r in rows)
            tot_n = sum(r["n_words"] for r in rows)
            print(f"[{i + 1}/{len(items)}] WER so far {100 * tot_e / max(tot_n, 1):.2f}%  ({sec:.1f}s)")
            if a.limit:
                print(f"   REF: {rn}\n   HYP: {hn}")

    out = a.out or os.path.join(RESULTS, f"{a.id}_{a.dataset}.csv")
    if a.limit:
        out = out.replace(".csv", f"_limit{a.limit}.csv")
    save_results(out, rows, {
        "model_id": a.id, "dataset": a.dataset, "ckpt": a.ckpt, "lm": a.lm, "lm_weight": a.lm_weight,
        "beam": a.beam, "gpu_decode": dec_dev != "cpu", "arch": arch, "params_M": round(n_params / 1e6, 1),
        "peak_gpu_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None,
        "total_seconds": round(time.time() - t_all, 1)})


if __name__ == "__main__":
    main()
