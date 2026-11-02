#!/usr/bin/env python
"""Auto-AVSR 在 LRS3 测试集上评估，直接读 parquet 的 96x96 ROI"""
import sys
sys.path.insert(0, "/root/autodl-tmp/code/auto_avsr")

import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio
from argparse import Namespace

from lightning import ModelModule
from datamodule.transforms import VideoTransform


def build_args():
    return Namespace(modality="video", pretrained_model_path=None, ctc_weight=0.1)


def frames_to_tensor(video_frames):
    frames = np.stack([np.asarray(f, dtype=np.uint8) for f in video_frames])  # T x 96 x 96
    t = torch.from_numpy(frames).unsqueeze(1)  # T x 1 x 96 x 96
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args_cli = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    args = build_args()
    ckpt = torch.load(args_cli.ckpt, map_location="cpu")
    # 有些 checkpoint 是嵌套的 dict，取出真正的 state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    # 去掉可能的前缀 (encoder. / model.)
    new_ckpt = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_ckpt[nk] = v
    ckpt = new_ckpt

    model = ModelModule(args)
    missing, unexpected = model.model.load_state_dict(ckpt, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) > 20:
        print("WARNING: 太多 missing keys，可能 checkpoint 结构不对")
        print("Sample missing:", missing[:5])
        print("Sample ckpt keys:", list(ckpt.keys())[:5])
    model.eval()
    model.to(device)
    print("Model loaded.")

    video_transform = VideoTransform(subset="test")

    parquet_files = sorted(glob.glob(os.path.join(args_cli.parquet_dir, "*.parquet")))
    print(f"Found {len(parquet_files)} parquet files")
    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")

    if args_cli.limit > 0:
        df = df.head(args_cli.limit)
        print(f"Limited to {len(df)} samples")

    results = []
    total_dist = 0
    total_words = 0

    for i, row in df.iterrows():
        ref = str(row["label"]).strip()
        try:
            video = frames_to_tensor(row["video"]).float()
            video = video_transform(video)
            video = video.to(device)
            with torch.no_grad():
                hyp = model(video)
            hyp = str(hyp).strip()
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            hyp = ""

        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()
        dist = torchaudio.functional.edit_distance(ref_words, hyp_words)
        total_dist += dist
        total_words += len(ref_words)

        results.append({"idx": i, "ref": ref, "hyp": hyp, "dist": dist, "n_words": len(ref_words)})

        if args_cli.limit > 0:
            print(f"[{i}] REF: {ref}")
            print(f"[{i}] HYP: {hyp}")
            print(f"[{i}] dist={dist}, words={len(ref_words)}")
            print("-" * 30)

        if (i + 1) % 50 == 0:
            cur_wer = 100.0 * total_dist / max(total_words, 1)
            print(f"  [{i+1}/{len(df)}] running WER = {cur_wer:.2f}%")

    wer = 100.0 * total_dist / max(total_words, 1)
    print("\n" + "=" * 40)
    print(f"  Samples: {len(df)}")
    print(f"  Total words: {total_words}")
    print(f"  Total edit distance: {total_dist}")
    print(f"  WER: {wer:.2f}%")
    print("=" * 40)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    pd.DataFrame(results).to_csv(args_cli.out, index=False)
    print(f"Saved to {args_cli.out}")

    summary = args_cli.out.replace(".csv", "_summary.txt")
    with open(summary, "w") as f:
        f.write(f"Model: {os.path.basename(args_cli.ckpt)}\n")
        f.write(f"Dataset: LRS3 test ({len(df)} samples)\n")
        f.write(f"WER: {wer:.2f}%\n")
        f.write(f"Total words: {total_words}, Edit distance: {total_dist}\n")
    print(f"Saved summary to {summary}")


if __name__ == "__main__":
    main()
