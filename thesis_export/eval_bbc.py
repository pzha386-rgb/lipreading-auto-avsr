#!/usr/bin/env python
"""Auto-AVSR 在 BBC (Tom Scott) 测试集上评估"""
import sys
sys.path.insert(0, "/root/autodl-tmp/code/auto_avsr")

import os, argparse, csv
import numpy as np
# 兼容 facexlib 老代码用的 np.float / np.int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
import cv2
import torch
import torchaudio
from argparse import Namespace
from lightning import ModelModule
from datamodule.transforms import VideoTransform
from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model


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


CROP_SIZE = 96

def extract_mouth_roi(frame_bgr, det_model, align_model):
    with torch.no_grad():
        bboxes = det_model.detect_faces(frame_bgr, 0.97)
    if bboxes is None or len(bboxes) == 0:
        return None
    bbox = bboxes[0][:4].astype(int)
    x1b, y1b, x2b, y2b = bbox
    # 裁出人脸区域喂给 alignment
    face_h, face_w = frame_bgr.shape[:2]
    x1b, y1b = max(0,x1b), max(0,y1b)
    x2b, y2b = min(face_w,x2b), min(face_h,y2b)
    face_crop = frame_bgr[y1b:y2b, x1b:x2b]
    if face_crop.size == 0:
        return None
    with torch.no_grad():
        lm = align_model.get_landmarks(face_crop)
    if lm is None or len(lm) == 0:
        return None
    # landmark 坐标是相对 face_crop 的，加回偏移
    lm = np.array(lm)
    if lm.ndim == 3:
        lm = lm[0]
    lm[:, 0] += x1b
    lm[:, 1] += y1b
    mouth_pts = lm[76:96]
    cx, cy = int(round(mouth_pts[:,0].mean())), int(round(mouth_pts[:,1].mean()))
    half = CROP_SIZE // 2
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = cx-half, cy-half, cx+half, cy+half
    pad_l, pad_t = max(0,-x1), max(0,-y1)
    pad_r, pad_b = max(0,x2-w), max(0,y2-h)
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t,pad_b,pad_l,pad_r, cv2.BORDER_CONSTANT, value=0)
    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def load_clip_rois(clip_path, det_model, align_model):
    cap = cv2.VideoCapture(clip_path)
    rois, last_valid = [], None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        roi = extract_mouth_roi(frame, det_model, align_model)
        if roi is None:
            roi = last_valid
        else:
            last_valid = roi
        if roi is not None:
            rois.append(roi)
    cap.release()
    return np.stack(rois) if rois else None


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
    print("facexlib detectors loaded.")

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
            roi = load_clip_rois(clip_path, det_model, align_model)
            if roi is None:
                raise RuntimeError("no ROI")
            video = torch.from_numpy(roi).unsqueeze(1).float()
            video = vt(video).to(device)
            with torch.no_grad():
                hyp = str(model(video)).strip()
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
