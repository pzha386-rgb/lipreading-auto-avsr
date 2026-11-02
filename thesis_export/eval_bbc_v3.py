#!/usr/bin/env python
"""Auto-AVSR BBC 评估 v3 — 手动 landmark 解码（复现 28.97%）"""
import sys
sys.path.insert(0, "/root/autodl-tmp/code/auto_avsr")

import os, argparse, csv
import numpy as np
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
import cv2
import torch
import torchaudio
import torchvision
from argparse import Namespace
from lightning import ModelModule
from datamodule.transforms import VideoTransform
from preparation.detectors.retinaface.video_process import VideoProcess
from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model, landmark_98_to_68


def convert_ckpt(ckpt):
    new = {}
    for k, v in ckpt.items():
        if k.startswith("encoder.frontend."):
            nk = k.replace("encoder.frontend.", "frontend.", 1)
        elif k.startswith("encoder.embed."):
            nk = k.replace("encoder.embed.0.", "proj_encoder.", 1)
        else:
            nk = k
        new[nk] = v
    return new


def decode_landmark(frame_rgb, bbox, align_model, margin=0.3):
    """手动 landmark 解码：crop+256 resize -> heatmap argmax + subpixel"""
    H_orig, W_orig = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half = max(x2-x1, y2-y1) * (1 + margin) / 2
    sx1, sy1 = max(0, int(cx-half)), max(0, int(cy-half))
    sx2, sy2 = min(W_orig, int(cx+half)), min(H_orig, int(cy+half))

    crop = frame_rgb[sy1:sy2, sx1:sx2]
    H_crop, W_crop = crop.shape[:2]
    if H_crop < 10 or W_crop < 10:
        return None

    crop_256 = cv2.resize(crop, (256, 256))
    crop_256_bgr = crop_256[..., ::-1]
    inp = torch.from_numpy(np.ascontiguousarray(
        crop_256_bgr.transpose((2, 0, 1)))).float()
    inp = inp.cuda().div(255).unsqueeze(0)

    with torch.no_grad():
        outputs, _ = align_model(inp)
    heatmap = outputs[-1][:, :-1, :, :]

    hm = heatmap[0].cpu().numpy()
    flat = hm.reshape(98, -1)
    idx = flat.argmax(axis=1)
    ys = idx // 64
    xs = idx % 64
    pred = np.stack([xs, ys], axis=1).astype(np.float32)

    for i in range(98):
        py, px = ys[i], xs[i]
        if 1 <= px <= 62 and 1 <= py <= 62:
            diff_x = hm[i, py, px+1] - hm[i, py, px-1]
            diff_y = hm[i, py+1, px] - hm[i, py-1, px]
            pred[i, 0] += np.sign(diff_x) * 0.25
            pred[i, 1] += np.sign(diff_y) * 0.25

    pred = pred * 4.0
    pred = pred * np.array([W_crop / 256, H_crop / 256])
    pred = pred + np.array([sx1, sy1])
    # 98 -> 68
    lm68 = landmark_98_to_68(pred)
    lm68 = np.array(lm68)
    if lm68.ndim == 3:
        lm68 = lm68[0]
    return lm68


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    args = Namespace(modality="video", pretrained_model_path=None, ctc_weight=0.1)
    ckpt = convert_ckpt(torch.load(a.ckpt, map_location="cpu"))
    model = ModelModule(args)
    missing, unexpected = model.model.load_state_dict(ckpt, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    model.eval().to(device)
    print("Auto-AVSR loaded.")

    det_model = init_detection_model('retinaface_resnet50', half=False, device=device)
    align_model = init_alignment_model('awing_fan', device=device)
    print("facexlib loaded.")

    video_process = VideoProcess(convert_gray=False)
    vt = VideoTransform(subset="test")

    rows = []
    with open(a.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if a.limit > 0:
        rows = rows[:a.limit]
    print(f"Total clips: {len(rows)}")

    import pandas as pd
    results, total_dist, total_words = [], 0, 0
    for i, r in enumerate(rows):
        clip_path = os.path.join(a.clips_dir, r["path"])
        ref = r["text"].strip()
        try:
            video = torchvision.io.read_video(clip_path, pts_unit="sec")[0].numpy()  # T,H,W,3 RGB
            landmarks = []
            for t in range(video.shape[0]):
                frame_rgb = video[t]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                with torch.no_grad():
                    bboxes = det_model.detect_faces(frame_bgr, 0.5)
                if bboxes is None or len(bboxes) == 0:
                    landmarks.append(None)
                    continue
                bbox = bboxes[0][:4]
                lm = decode_landmark(frame_rgb, bbox, align_model, margin=0.3)
                landmarks.append(lm)
            seq = video_process(video, landmarks)
            if seq is None:
                raise RuntimeError("VideoProcess None")
            seq = np.array(seq)
            if seq.ndim == 4 and seq.shape[-1] == 3:
                seq = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq])
            vid = torch.from_numpy(seq).unsqueeze(1).float()
            vid = vt(vid).to(device)
            with torch.no_grad():
                hyp = str(model(vid)).strip()
        except Exception as e:
            print(f"[{i}] {r['clip_id']} ERROR: {e}")
            hyp = ""
        rw, hw = ref.lower().split(), hyp.lower().split()
        d = torchaudio.functional.edit_distance(rw, hw)
        total_dist += d; total_words += len(rw)
        results.append({"clip_id": r["clip_id"], "ref": ref, "hyp": hyp, "dist": d, "n_words": len(rw)})
        if a.limit > 0 or (i+1) % 10 == 0:
            print(f"[{i+1}/{len(rows)}] {r['clip_id']} WER={100.0*total_dist/max(total_words,1):.2f}%")
            if a.limit > 0:
                print(f"   REF: {ref}")
                print(f"   HYP: {hyp}")

    wer = 100.0*total_dist/max(total_words,1)
    print(f"\n{'='*40}\nBBC clips: {len(rows)}  Words: {total_words}\nWER: {wer:.2f}%\n{'='*40}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    with open(a.out.replace(".csv","_summary.txt"),"w") as f:
        f.write(f"Model: {os.path.basename(a.ckpt)}\nDataset: BBC Tom Scott ({len(rows)} clips)\nWER: {wer:.2f}%\n")
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
