# -*- coding: utf-8 -*-
"""
adapt_autoavsr.py  -  fine-tune Auto-AVSR (T1) on N minutes of Tom Scott

- model, weights, transforms, tokenizer: the same as your evaluation (load_model, VideoTransform, TextTransform)
- loss: the model's own CTC + attention loss
- data: N minutes from the TRAIN videos only. The clip order is fixed (seed), so
        the 5-min set is inside the 15-min set, which is inside the 30-min set ... (fair curve)
- early stopping on the VAL videos (validation loss)
- saves best.pth in the same format as the original weights -> eval_autoavsr_nbest.py reads it directly

Put in /root/autodl-tmp/vsr_bench/ and run there:
  python adapt_autoavsr.py --minutes 5 --check                 # 3 steps only, to test
  python adapt_autoavsr.py --minutes 30
Output: /root/autodl-tmp/adapt/autoavsr_m<minutes>/best.pth + history.json
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CACHE                                    # noqa: E402
from eval_autoavsr_cache import load_model                  # noqa: E402  (also adds auto_avsr to path)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/root/autodl-tmp/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth")
ap.add_argument("--convert-fn", default="")
ap.add_argument("--minutes", type=float, required=True, help="minutes of training speech (0 = all)")
ap.add_argument("--name", default="tomscott")
ap.add_argument("--out", default="")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--weight-decay", type=float, default=0.1)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--max-frames", type=int, default=2000, help="padded frames per batch (memory)")
ap.add_argument("--no-freeze", action="store_true", help="also train the visual frontend (ResNet)")
ap.add_argument("--freeze-layers", type=int, default=-1,
                help="freeze all conformer layers except the last N (-1 = train all non-frontend layers)")
ap.add_argument("--anchor", type=float, default=0.01, help="L2 penalty toward original weights")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--check", action="store_true")
a = ap.parse_args()

OUT = a.out or f"/root/autodl-tmp/adapt/autoavsr_m{a.minutes:g}"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(a.seed)
device = "cuda"

# ---------------------------------------------------------------- model
mm = load_model(argparse.Namespace(ckpt=a.ckpt, convert_fn=a.convert_fn), device)
model, tt = mm.model, mm.text_transform
from datamodule.transforms import VideoTransform          # noqa: E402
vt_train, vt_test = VideoTransform(subset="train"), VideoTransform(subset="test")
freeze = not a.no_freeze
if freeze:
    for p in model.frontend.parameters():
        p.requires_grad = False
if a.freeze_layers > 0:
    # freeze all encoder layers except the last N
    enc_layers = [m for m in model.encoder.encoders if hasattr(m, 'parameters')]
    n_freeze = max(0, len(enc_layers) - a.freeze_layers)
    for layer in enc_layers[:n_freeze]:
        for p in layer.parameters():
            p.requires_grad = False
    # also freeze proj_encoder (the linear projection before the conformer)
    if hasattr(model, 'proj_encoder'):
        for p in model.proj_encoder.parameters():
            p.requires_grad = False
    print(f"encoder: {len(enc_layers)} layers total, last {a.freeze_layers} trainable, first {n_freeze} frozen")
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_train / 1e6:.1f} M   (frontend {'frozen' if freeze else 'trained'})")

# ---------------------------------------------------------------- data
try:
    import inflect
    _P = inflect.engine()
except ImportError:
    _P = None


def to_label(text):
    """LRS3 style: UPPER CASE, numbers as words, keep apostrophes"""
    t = str(text)
    if _P:
        t = re.sub(r"\d+(\.\d+)?", lambda m: _P.number_to_words(m.group(), andword=""), t)
    t = t.upper().replace("-", " ").replace("\u2019", "'")
    t = re.sub(r"[^A-Z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


src = os.path.join(CACHE, a.name)
man = pd.read_csv(os.path.join(src, "manifest_frames.csv"), keep_default_na=False)
split = json.load(open(os.path.join(CACHE, f"{a.name}_split.json")))


def make_items(df):
    items = []
    for _, r in df.iterrows():
        lab = to_label(r["text"])
        ids = tt.tokenize(lab)
        if len(ids) == 0 or r["frames"] < 2 * len(ids):      # too short for CTC
            continue
        items.append((os.path.join(src, r["npy"]), ids, int(r["frames"])))
    return items


pool = man[man["video_id"].isin(split["train"])].sample(frac=1.0, random_state=a.seed)   # fixed order
if a.minutes > 0:
    keep = (pool["frames"].cumsum() / 25 / 60) <= a.minutes
    keep.iloc[0] = True
    pool = pool[keep]
train_items = make_items(pool)
val_items = make_items(man[man["video_id"].isin(split["val"])])
tr_min = sum(i[2] for i in train_items) / 25 / 60
print(f"train: {len(train_items)} clips, {tr_min:.1f} min   val: {len(val_items)} clips")


def batches(items, shuffle):
    idx = sorted(range(len(items)), key=lambda i: items[i][2])
    out, cur, cur_max = [], [], 0
    for i in idx:
        m = max(cur_max, items[i][2])
        if cur and m * (len(cur) + 1) > a.max_frames:
            out.append(cur)
            cur, m = [], items[i][2]
        cur.append(i)
        cur_max = m
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(items, ids, train):
    vt = vt_train if train else vt_test
    vids = [vt(torch.from_numpy(np.load(items[i][0])).unsqueeze(1).float()) for i in ids]
    T = max(v.shape[0] for v in vids)
    x = torch.zeros((len(vids), T) + tuple(vids[0].shape[1:]))
    for k, v in enumerate(vids):
        x[k, :v.shape[0]] = v
    lengths = torch.tensor([v.shape[0] for v in vids])
    L = max(len(items[i][1]) for i in ids)
    y = torch.full((len(ids), L), -1, dtype=torch.long)
    for k, i in enumerate(ids):
        y[k, :len(items[i][1])] = items[i][1]
    return x.to(device), lengths.to(device), y.to(device)


def step(x, lengths, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, lengths, y)
    loss = out[0] if isinstance(out, (tuple, list)) else out
    return loss.float()


@torch.no_grad()
def val_loss():
    model.eval()
    tot, n = 0.0, 0
    for b in batches(val_items, False):
        tot += step(*collate(val_items, b, False)).item() * len(b)
        n += len(b)
    return tot / max(n, 1)


# ---------------------------------------------------------------- train
steps_per_epoch = len(batches(train_items, False))
total = steps_per_epoch * a.epochs
warm = max(1, int(0.1 * total))
# save original weights for regularization
orig = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

best, bad, hist, gstep, t0 = float("inf"), 0, [], 0, time.time()
v0 = val_loss()
print(f"epoch 0  val loss {v0:.3f}  (before adaptation)")
hist.append({"epoch": 0, "val_loss": v0})
for ep in range(1, a.epochs + 1):
    model.train()
    if freeze:
        model.frontend.eval()                                  # keep BatchNorm statistics fixed
    tl, tn = 0.0, 0
    for b in batches(train_items, True):
        loss = step(*collate(train_items, b, True))
        # anchor regularization: penalize drifting from original weights
        if a.anchor > 0:
            for n, p in model.named_parameters():
                if p.requires_grad and n in orig:
                    loss = loss + a.anchor * ((p - orig[n]) ** 2).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        gstep += 1
        tl, tn = tl + loss.item() * len(b), tn + len(b)
        if a.check:
            print(f"  check step {gstep}: loss {loss.item():.3f}   gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            if gstep >= 3:
                raise SystemExit("check OK - training works")
    vl = val_loss()
    hist.append({"epoch": ep, "train_loss": tl / max(tn, 1), "val_loss": vl, "minutes": (time.time() - t0) / 60})
    flag = ""
    if vl < best:
        best, bad, flag = vl, 0, "  * saved"
        torch.save(model.state_dict(), os.path.join(OUT, "best.pth"))
    else:
        bad += 1
    print(f"epoch {ep:2d}  train {tl / max(tn, 1):.3f}  val {vl:.3f}{flag}   ({(time.time() - t0) / 60:.1f} min)")
    json.dump({"args": vars(a), "train_min": tr_min, "history": hist}, open(os.path.join(OUT, "history.json"), "w"), indent=2)
    if bad >= a.patience:
        print("early stop")
        break

print(f"\nDONE  best val loss {best:.3f} (start {v0:.3f})  ->  {OUT}/best.pth")
print(f"evaluate:  python -u eval_autoavsr_nbest.py --id A{a.minutes:g} --ckpt {OUT}/best.pth --dataset bbc")
