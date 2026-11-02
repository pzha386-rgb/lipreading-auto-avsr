#!/usr/bin/env python
"""转换 checkpoint key 命名 + 跑 Auto-AVSR 评估"""
import sys
sys.path.insert(0, "/root/autodl-tmp/code/auto_avsr")

import os, argparse, glob
import numpy as np
import pandas as pd
import torch
import torchaudio
from argparse import Namespace
from lightning import ModelModule
from datamodule.transforms import VideoTransform


def convert_ckpt(ckpt):
    """把 espnet 原生命名转成 auto_avsr E2E 命名"""
    new = {}
    for k, v in ckpt.items():
        if k.startswith("encoder.frontend."):
            nk = k.replace("encoder.frontend.", "frontend.", 1)
        elif k.startswith("encoder.embed."):
            # encoder.embed.0 -> proj_encoder
            nk = k.replace("encoder.embed.0.", "proj_encoder.", 1)
        else:
            nk = k  # encoder.* / decoder.* / ctc.* 保持
        new[nk] = v
    return new


def frames_to_tensor(video_frames):
    # 每帧是 (96,) 的 object 数组，每个元素又是 (96,) 数组，需要逐层堆叠
    frames = []
    for f in video_frames:
        rows = [np.asarray(r, dtype=np.uint8) for r in f]
        frames.append(np.stack(rows))  # 96 x 96
    arr = np.stack(frames)  # T x 96 x 96
    return torch.from_numpy(arr).unsqueeze(1)  # T x 1 x 96 x 96


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    args = Namespace(modality="video", pretrained_model_path=None, ctc_weight=0.1)
    ckpt = torch.load(a.ckpt, map_location="cpu")
    ckpt = convert_ckpt(ckpt)

    model = ModelModule(args)
    missing, unexpected = model.model.load_state_dict(ckpt, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if len(missing) > 5:
        print("Missing 示例:", missing[:5])
        print("Unexpected 示例:", unexpected[:5])
        print("警告: 仍有较多 key 不匹配，检查映射")
    model.eval().to(device)
    print("Model loaded.")

    vt = VideoTransform(subset="test")
    files = sorted(glob.glob(os.path.join(a.parquet_dir, "*.parquet")))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    print(f"Total samples: {len(df)}")
    if a.limit > 0:
        df = df.head(a.limit)

    results, total_dist, total_words = [], 0, 0
    for i, row in df.iterrows():
        ref = str(row["label"]).strip()
        try:
            v = frames_to_tensor(row["video"]).float()
            v = vt(v).to(device)
            with torch.no_grad():
                hyp = str(model(v)).strip()
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            hyp = ""
        rw, hw = ref.lower().split(), hyp.lower().split()
        d = torchaudio.functional.edit_distance(rw, hw)
        total_dist += d; total_words += len(rw)
        results.append({"idx": i, "ref": ref, "hyp": hyp, "dist": d, "n_words": len(rw)})
        if a.limit > 0:
            print(f"[{i}] REF: {ref}")
            print(f"[{i}] HYP: {hyp}")
            print("-"*30)
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}] WER={100.0*total_dist/max(total_words,1):.2f}%")

    wer = 100.0*total_dist/max(total_words,1)
    print(f"\n{'='*40}\nSamples: {len(df)}  Words: {total_words}\nWER: {wer:.2f}%\n{'='*40}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    with open(a.out.replace(".csv","_summary.txt"),"w") as f:
        f.write(f"Model: {os.path.basename(a.ckpt)}\nDataset: LRS3 test ({len(df)})\nWER: {wer:.2f}%\n")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
