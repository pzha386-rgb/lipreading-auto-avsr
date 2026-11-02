#!/usr/bin/env python
"""Auto-AVSR BBC 评估 v2 — 用官方 VideoProcess 做仿射对齐（复现 28.97%）"""
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


def detect_landmarks_68(frame_bgr, det_model, align_model):
    """返回 68 点 landmark，失败返回 None"""
    with torch.no_grad():
        bboxes = det_model.detect_faces(frame_bgr, 0.5)
    if bboxes is None or len(bboxes) == 0:
        return None
    bbox = bboxes[0][:4].astype(int)
    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w,x2), min(h,y2)
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    with torch.no_grad():
        lm98 = align_model.get_landmarks(face)
    if lm98 is None or len(lm98) == 0:
        return None
    lm98 = np.array(lm98)
    if lm98.ndim == 3:
        lm98 = lm98[0]
    lm98[:, 0] += x1
    lm98[:, 1] += y1
    try:
        lm68 = landmark_98_to_68(lm98)
    except Exception:
        lm68 = landmark_98_to_68(lm98[np.newaxis, ...])
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
                frame_bgr = cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR)
                lm = detect_landmarks_68(frame_bgr, det_model, align_model)
                landmarks.append(lm)
            # 官方 VideoProcess: 输入 RGB video + 68 landmarks
            seq = video_process(video, landmarks)
            if seq is None:
                raise RuntimeError("VideoProcess returned None")
            seq = np.array(seq)
            # convert_gray=False 输出 RGB, 转灰度
            if seq.ndim == 4 and seq.shape[-1] == 3:
                seq = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq])
            vid = torch.from_numpy(seq).unsqueeze(1).float()  # T x 1 x 96 x 96
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
