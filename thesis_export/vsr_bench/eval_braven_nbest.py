# -*- coding: utf-8 -*-
"""
eval_braven_nbest.py  -  BRAVEn with N-best + confidence scores

Same as eval_raven_family.py but keeps top-k hypotheses and their scores,
outputs confA_logp / confA_margin / confA_nbest (same format as eval_autoavsr_nbest.py).

Run in /root/autodl-tmp/vsr_bench:
  # B2 (BRAVEn low +ST +LM) on Tom Scott test - limit test
  python -u eval_braven_nbest.py --id B2 \
    --ckpt /root/autodl-tmp/checkpoints/raven/vsr_prelrs3vox2avs_large_ftlrs3trainvalvox2avs_selftrain_braven.pth \
    --lm /root/autodl-tmp/checkpoints/raven/rnnlm.model.best \
    --dataset bbc --limit 5

  # full runs
  python -u eval_braven_nbest.py --id B2 --ckpt ... --lm ... --dataset tomscott_test
  python -u eval_braven_nbest.py --id B2 --ckpt ... --lm ... --dataset bbc
  python -u eval_braven_nbest.py --id B2 --ckpt ... --lm ... --dataset lrs3_300
  python -u eval_braven_nbest.py --id B2 --ckpt ... --lm ... --dataset tomscott
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/autodl-tmp/day1")

from common import read_manifest, score                  # noqa: E402
from eval_raven_family import build, prep, to_text        # noqa: E402
from conf_utils import nbest_features, normalize as conf_normalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--id", default="B2")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--lm", default="")
ap.add_argument("--lm-weight", type=float, default=None)
ap.add_argument("--beam", type=int, default=40)
ap.add_argument("--size", default="large")
ap.add_argument("--k", type=int, default=5, help="N-best list size")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--outdir", default="/root/autodl-tmp/day1/conf")
# needed by build() but not used here
ap.add_argument("--gpu-decode", action="store_true")
ap.add_argument("--out", default="")
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"===== {a.id} N-best on {a.dataset} =====")
cfg, learner, arch = build(a)
learner.model.encoder = learner.model.encoder.to(device)

from utils import UNIGRAM1000_LIST  # noqa: E402

items = read_manifest(a.dataset)
if a.limit:
    items = items[:a.limit]

import pandas as pd
tag = f"{a.id}_{a.dataset}" + (f"_limit{a.limit}" if a.limit else "")
rows, tot_e, tot_n, t_all = [], 0, 0, time.time()
fj = open(os.path.join(a.outdir, f"{tag}_nbest.jsonl"), "w", encoding="utf-8")

for i, (uid, ref, npy) in enumerate(items):
    nbest_out, sec = [], 0.0
    try:
        v = prep(np.load(npy)).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            feat, _ = learner.model.encoder(v, None)
            hyps = learner.beam_search(x=feat.squeeze(0).cpu(),
                                       maxlenratio=cfg.decode.maxlenratio,
                                       minlenratio=cfg.decode.minlenratio)
        sec = time.perf_counter() - t0

        for h in hyps[:a.k]:
            d = h.asdict()
            # decode text
            yseq = d["yseq"][1:]  # remove SOS
            text_tokens = [UNIGRAM1000_LIST[int(t)] for t in yseq if int(t) < len(UNIGRAM1000_LIST)]
            text = "".join(text_tokens).replace("▁", " ").replace("<space>", " ").replace("<eos>", "").strip()
            sc = d.get("scores", {})
            nbest_out.append({
                "text": text,
                "score": float(d["score"]),
                "n_tok": max(len(yseq), 1),
                "dec": float(sc.get("decoder", 0.0)),
                "ctc": float(sc.get("ctc", 0.0)),
            })
    except Exception as e:
        print(f"[{i}] {uid} ERROR: {e}")
        if i == 0:
            import traceback
            traceback.print_exc()

    hyp = nbest_out[0]["text"] if nbest_out else ""
    rn, hn, err, n = score(ref, hyp)
    tot_e, tot_n = tot_e + err, tot_n + n
    feats = nbest_features(nbest_out)
    rows.append({"id": uid, "ref": rn, "hyp": hn, "errors": err, "n_words": n,
                 "seconds": round(sec, 3), **feats})
    fj.write(json.dumps({"id": uid, "ref": ref, "nbest": nbest_out, **feats}, ensure_ascii=False) + "\n")

    if a.limit or (i + 1) % 20 == 0:
        print(f"[{i + 1}/{len(items)}] WER so far {100 * tot_e / max(tot_n, 1):.2f}%  ({sec:.2f}s)  "
              f"logp={feats['confA_logp']:.3f} margin={feats['confA_margin']:.3f} nbest={feats['confA_nbest']:.2f}")
        if a.limit:
            print(f"   REF: {rn}\n   HYP: {hn}")
            for j, h in enumerate(nbest_out[1:3], 2):
                print(f"   #{j}: {h['text']}   (score {h['score']:.2f})")
fj.close()

out = os.path.join(a.outdir, f"{tag}_confA.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nfinal WER {100 * tot_e / max(tot_n, 1):.2f}%   total {time.time() - t_all:.0f}s")
print(f"saved {out}")
