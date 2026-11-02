# -*- coding: utf-8 -*-
"""
lora_adapt.py  -  LoRA fine-tuning of Auto-AVSR (T1) on Tom Scott

Unlike adapt_autoavsr.py (which failed due to catastrophic forgetting),
LoRA only adds tiny rank-r matrices alongside the attention projections.
The original model weights are COMPLETELY FROZEN.

Typical LoRA parameters: rank=8, ~0.5M trainable (vs 239M total).
This makes catastrophic forgetting almost impossible.

Run in /root/autodl-tmp/vsr_bench:
  python lora_adapt.py --minutes 30 --check           # test 3 steps
  python lora_adapt.py --minutes 30                   # train
  python lora_adapt.py --minutes 120
Output: /root/autodl-tmp/adapt_lora/m<minutes>/lora.pth + merged.pth
  merged.pth can be used directly with eval_autoavsr_nbest.py
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/autodl-tmp/day1")
from common import CACHE
from eval_autoavsr_cache import load_model
from conf_utils import normalize as conf_normalize

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/root/autodl-tmp/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth")
ap.add_argument("--convert-fn", default="")
ap.add_argument("--minutes", type=float, required=True)
ap.add_argument("--name", default="tomscott")
ap.add_argument("--out", default="")
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=float, default=16.0, help="LoRA scaling = alpha / rank")
ap.add_argument("--lr", type=float, default=5e-4, help="LoRA layers can use higher LR")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--max-frames", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--check", action="store_true")
a = ap.parse_args()

OUT = a.out or f"/root/autodl-tmp/adapt_lora/m{a.minutes:g}"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(a.seed)
device = "cuda"


# ---- LoRA layer
class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adapter. Original weight is frozen."""
    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        d_in, d_out = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def apply_lora(model, rank, alpha, target_modules=("linear_q", "linear_k", "linear_v", "linear_out")):
    """Replace target Linear layers in the encoder with LoRA versions"""
    count = 0
    for name, module in model.named_modules():
        for attr in target_modules:
            if hasattr(module, attr):
                orig = getattr(module, attr)
                if isinstance(orig, nn.Linear):
                    setattr(module, attr, LoRALinear(orig, rank, alpha))
                    count += 1
    return count


# ---- model
mm = load_model(argparse.Namespace(ckpt=a.ckpt, convert_fn=a.convert_fn), device)
model = mm.model
tt = mm.text_transform
from datamodule.transforms import VideoTransform
vt_train, vt_test = VideoTransform(subset="train"), VideoTransform(subset="test")

# freeze everything
for p in model.parameters():
    p.requires_grad = False

# apply LoRA to encoder attention layers
n_lora = apply_lora(model, a.rank, a.alpha)
model = model.to(device)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA applied to {n_lora} layers, rank={a.rank}, trainable params: {n_train / 1e6:.2f} M "
      f"(vs {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M total)")

# ---- data (same as adapt_autoavsr.py)
try:
    import inflect
    _P = inflect.engine()
except ImportError:
    _P = None


def to_label(text):
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
        if len(ids) == 0 or r["frames"] < 2 * len(ids):
            continue
        items.append((os.path.join(src, r["npy"]), ids, int(r["frames"])))
    return items


import random
pool = man[man["video_id"].isin(split["train"])].sample(frac=1.0, random_state=a.seed)
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


# ---- train (only LoRA params)
lora_params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(lora_params, lr=a.lr, weight_decay=0.01)
total_steps = len(batches(train_items, False)) * a.epochs
warm = max(1, int(0.1 * total_steps))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))

best, bad, hist, gstep, t0 = float("inf"), 0, [], 0, time.time()
v0 = val_loss()
print(f"epoch 0  val loss {v0:.3f}  (before LoRA)")
hist.append({"epoch": 0, "val_loss": v0})

for ep in range(1, a.epochs + 1):
    model.train()
    model.frontend.eval()
    tl, tn = 0.0, 0
    for b in batches(train_items, True):
        loss = step(*collate(train_items, b, True))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 10.0)
        opt.step()
        sched.step()
        gstep += 1
        tl, tn = tl + loss.item() * len(b), tn + len(b)
        if a.check:
            print(f"  check step {gstep}: loss {loss.item():.3f}   gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            if gstep >= 3:
                raise SystemExit("check OK - LoRA training works")
    vl = val_loss()
    hist.append({"epoch": ep, "train_loss": tl / max(tn, 1), "val_loss": vl, "minutes": (time.time() - t0) / 60})
    flag = ""
    if vl < best:
        best, bad, flag = vl, 0, "  * saved"
        # save only LoRA weights
        lora_sd = {n: p.data for n, p in model.named_parameters() if p.requires_grad}
        torch.save(lora_sd, os.path.join(OUT, "lora.pth"))
    else:
        bad += 1
    print(f"epoch {ep:2d}  train {tl / max(tn, 1):.3f}  val {vl:.3f}{flag}   ({(time.time() - t0) / 60:.1f} min)")
    json.dump({"args": vars(a), "train_min": tr_min, "n_lora_params": n_train, "history": hist},
              open(os.path.join(OUT, "history.json"), "w"), indent=2)
    if bad >= a.patience:
        print("early stop")
        break

# ---- merge LoRA into original weights -> merged.pth (standard format, works with eval_autoavsr_nbest.py)
print("\nmerging LoRA into original weights...")
orig_sd = torch.load(a.ckpt, map_location="cpu")
if isinstance(orig_sd, dict) and "state_dict" in orig_sd:
    orig_sd = orig_sd["state_dict"]
# reload best LoRA
lora_sd = torch.load(os.path.join(OUT, "lora.pth"), map_location="cpu")
# apply LoRA: W_new = W_orig + scale * A @ B
for name, param in lora_sd.items():
    if name.endswith(".lora_A"):
        base = name[:-len(".lora_A")]
        A = param
        B = lora_sd[base + ".lora_B"]
        scale = a.alpha / a.rank
        # find the original weight key
        orig_key = base + ".original.weight"
        # the key in orig_sd might not have ".original"
        sd_key = base.replace(".original", "") + ".weight"
        if sd_key not in orig_sd:
            # try with the full path
            sd_key = orig_key.replace(".original", "")
        if sd_key in orig_sd:
            orig_sd[sd_key] = orig_sd[sd_key] + (A @ B).T * scale
            # note: Linear stores weight as (out, in), and our LoRA computes x @ A @ B
            # so the delta is (A @ B)^T = B^T @ A^T, shape (out, in) -- wait let me think
            # x @ A @ B has shape (batch, out) if A is (in, rank) and B is (rank, out)
            # Linear(x) = x @ W^T + bias, where W is (out, in)
            # So the delta to W should be (A @ B)^T = B^T @ A^T, which has shape (out, in) ✓

# actually, let me reconsider the shapes:
# LoRALinear.forward: original(x) + (x @ A @ B) * scale
# original(x) = x @ W.T  (W is out x in)
# x @ A @ B: x is (..., in), A is (in, rank), B is (rank, out) -> result is (..., out)
# So total = x @ (W.T + scale * A @ B) = x @ (W + scale * (A @ B).T).T
# Therefore W_new = W + scale * (A @ B).T = W + scale * B.T @ A.T
# But we already have orig_sd[key] which is W (out x in)
# delta = scale * B.T @ A.T = scale * (A @ B).T

# Let me redo the merge correctly
print("re-merging with correct transpose...")
orig_sd2 = torch.load(a.ckpt, map_location="cpu")
if isinstance(orig_sd2, dict) and "state_dict" in orig_sd2:
    orig_sd2 = orig_sd2["state_dict"]
merged = 0
for name in lora_sd:
    if not name.endswith(".lora_A"):
        continue
    base = name[:-len(".lora_A")]
    A = lora_sd[base + ".lora_A"]  # (in, rank)
    B = lora_sd[base + ".lora_B"]  # (rank, out)
    delta = (A @ B).T * (a.alpha / a.rank)  # (out, in)
    sd_key = base.replace(".original", "") + ".weight"
    if sd_key in orig_sd2:
        orig_sd2[sd_key] = orig_sd2[sd_key] + delta
        merged += 1
torch.save(orig_sd2, os.path.join(OUT, "merged.pth"))
print(f"merged {merged} LoRA layers -> {OUT}/merged.pth")
print(f"\nDONE  best val loss {best:.3f} (start {v0:.3f})")
print(f"evaluate:  python -u eval_autoavsr_nbest.py --id L{a.minutes:g} --ckpt {OUT}/merged.pth --dataset tomscott_test")
