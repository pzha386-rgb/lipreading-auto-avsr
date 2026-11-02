# -*- coding: utf-8 -*-
"""
USR 2.0 评测（U1–U4、U1g–U4g、U4b10；读 ROI 缓存）

和官方 demo.py 一样：E2E 模型 + BatchBeamSearch（在 GPU 上），只用视频（modality = v）
不同之处：不在这里检测人脸，而是读统一的 ROI 缓存，保证和其他模型看到的嘴部画面完全相同

运行：
  python eval_usr2.py --id U4 --ckpt /root/autodl-tmp/checkpoints/new/U4_usr2_high_huge.pth \
      --backbone resnet_transformer_huge --dataset bbc --limit 5
  # 快速解码
  python eval_usr2.py ... --beam 1 --ctc-weight 0
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

from common import RESULTS, ROOT, read_manifest, save_results, score

CODE_USR2 = os.environ.get("CODE_USR2", f"{ROOT}/code/usr2")
sys.path.insert(0, CODE_USR2)

CROP, MEAN, STD = 88, 0.421, 0.165     # 和 demo.py 的 build_video_transform 一样


def prep(arr):
    """T x 96 x 96 uint8 灰度 -> [1, T, 88, 88]"""
    arr = arr.astype(np.float32) / 255.0
    T, H, W = arr.shape
    i, j = (H - CROP) // 2, (W - CROP) // 2
    arr = (arr[:, i:i + CROP, j:j + CROP] - MEAN) / STD
    return torch.from_numpy(arr).unsqueeze(0)


def load_cfg(backbone):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("len", len, replace=True)
    avail = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{CODE_USR2}/conf/model/backbone/*.yaml"))
    if backbone not in avail:
        raise SystemExit(f"backbone '{backbone}' 不存在。可用的: {avail}")
    with initialize_config_dir(config_dir=f"{CODE_USR2}/conf", version_base="1.3"):
        return compose(config_name="config", overrides=[f"model/backbone={backbone}"])


def load_model(cfg, ckpt_path, device):
    """和 demo.py 的 load_model 完全一样"""
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from utils.utils import UNIGRAM1000_LIST
    model = E2E(len(UNIGRAM1000_LIST), cfg.model.backbone)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in ckpt):
        ckpt = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt.items()}
    if any(k.startswith("model.backbone.") for k in ckpt):
        ckpt = {k.replace("model.backbone.", "", 1): v for k, v in ckpt.items() if k.startswith("model.backbone.")}
    model.load_state_dict(ckpt)
    return model.eval().to(device)


def build_beam_search(model, beam, ctc_weight, penalty):
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus
    from utils.utils import UNIGRAM1000_LIST
    tl = UNIGRAM1000_LIST
    scorers = model.scorers()
    scorers["length_bonus"] = LengthBonus(len(tl))
    weights = dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, length_bonus=penalty)
    return BatchBeamSearch(beam_size=beam, vocab_size=len(tl), weights=weights, scorers=scorers,
                           sos=len(tl) - 1, eos=len(tl) - 1, token_list=tl,
                           pre_beam_score_key=None if ctc_weight == 1.0 else "decoder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", required=True, help="例如 resnet_transformer_large")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--beam", type=int, default=None, help="默认用官方配置")
    ap.add_argument("--ctc-weight", type=float, default=None, help="默认用官方配置")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== {a.id} on {a.dataset} =====")
    cfg = load_cfg(a.backbone)
    beam = a.beam if a.beam is not None else int(cfg.decode.beam_size)
    ctc = a.ctc_weight if a.ctc_weight is not None else float(cfg.decode.ctc_weight)
    print(f"官方解码配置: beam={cfg.decode.beam_size} ctc={cfg.decode.ctc_weight} penalty={cfg.decode.penalty}"
          f"  -> 本次使用 beam={beam} ctc={ctc}")

    model = load_model(cfg, a.ckpt, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params / 1e6:.1f} M")
    bs = build_beam_search(model, beam, ctc, float(cfg.decode.penalty))
    bs.to(device)

    from espnet.asr.asr_utils import parse_hypothesis
    from utils.utils import UNIGRAM1000_LIST

    items = read_manifest(a.dataset)
    if a.limit:
        items = items[:a.limit]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, t_all = [], time.time()
    for i, (uid, ref, npy) in enumerate(items):
        hyp, sec = "", 0.0
        try:
            v = prep(np.load(npy)).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                feat = model.encoder(xs_v=v)
                hyps = bs(x=feat.squeeze(0), modality="v",
                          maxlenratio=cfg.decode.maxlenratio, minlenratio=cfg.decode.minlenratio)
            text, _, _, _ = parse_hypothesis(hyps[0].asdict(), UNIGRAM1000_LIST)
            hyp = text.replace("<eos>", "").replace("\u2581", " ").strip()
            if device.type == "cuda":
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
        "model_id": a.id, "dataset": a.dataset, "ckpt": a.ckpt, "backbone": a.backbone,
        "beam": beam, "ctc_weight": ctc, "params_M": round(n_params / 1e6, 1),
        "peak_gpu_MB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else None,
        "total_seconds": round(time.time() - t_all, 1)})


if __name__ == "__main__":
    main()
