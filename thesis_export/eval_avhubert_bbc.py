#!/usr/bin/env python
"""AV-HuBERT (MuAViC-en) 在 BBC 上评估 - 纯视觉"""
import os, sys, argparse, csv, re, glob
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, "/root/autodl-tmp/code/AV-HuBERT-S2S")
sys.path.insert(0, "/root/autodl-tmp/code/auto_avsr")

import numpy as np
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'bool'): np.bool = bool
import cv2
import torch
import torch.nn.functional as F
import torchaudio
import torchvision
import inflect

from src.model.avhubert2text import AV2TextForConditionalGeneration
from transformers import Speech2TextTokenizer
from preparation.detectors.retinaface.video_process import VideoProcess
from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model, landmark_98_to_68

p = inflect.engine()
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    def n2w(m):
        try: return p.number_to_words(m.group(), andword='').replace('-',' ').replace(',',' ')
        except: return m.group()
    text = re.sub(r'\d+', n2w, text)
    return re.sub(r'\s+', ' ', text).strip()

IMAGE_CROP, IMAGE_MEAN, IMAGE_STD = 88, 0.421, 0.165

def center_crop(frames, size):
    T, H, W = frames.shape
    i, j = (H-size)//2, (W-size)//2
    return frames[:, i:i+size, j:j+size]

def detect_lm68(frame_bgr, det, align):
    with torch.no_grad():
        bboxes = det.detect_faces(frame_bgr, 0.5)
    if bboxes is None or len(bboxes)==0: return None
    b = bboxes[0][:4].astype(int)
    x1,y1,x2,y2 = b; h,w = frame_bgr.shape[:2]
    x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    face = frame_bgr[y1:y2, x1:x2]
    if face.size==0: return None
    with torch.no_grad():
        lm98 = align.get_landmarks(face)
    if lm98 is None or len(lm98)==0: return None
    lm98 = np.array(lm98)
    if lm98.ndim==3: lm98 = lm98[0]
    lm98[:,0]+=x1; lm98[:,1]+=y1
    lm68 = landmark_98_to_68(lm98)
    lm68 = np.array(lm68)
    if lm68.ndim==3: lm68=lm68[0]
    return lm68

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    print("加载 AV-HuBERT...")
    model = AV2TextForConditionalGeneration.from_pretrained(
        "nguyenvulebinh/AV-HuBERT-MuAViC-en", cache_dir="/root/autodl-tmp/checkpoints/avhubert_s2s")
    tokenizer = Speech2TextTokenizer.from_pretrained(
        "nguyenvulebinh/AV-HuBERT-MuAViC-en", cache_dir="/root/autodl-tmp/checkpoints/avhubert_s2s")
    model = model.cuda().eval()
    det = init_detection_model('retinaface_resnet50', half=False, device='cuda')
    align = init_alignment_model('awing_fan', device='cuda')
    vp = VideoProcess(convert_gray=True)
    print("就绪")

    rows = []
    with open(a.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f): rows.append(r)
    if a.limit>0: rows = rows[:a.limit]
    print(f"clips: {len(rows)}")

    import pandas as pd
    results, raw_d,raw_w,norm_d,norm_w = [],0,0,0,0
    for i, r in enumerate(rows):
        ref = r["text"].strip()
        try:
            video = torchvision.io.read_video(os.path.join(a.clips_dir, r["path"]), pts_unit="sec")[0].numpy()
            lms = []
            for t in range(video.shape[0]):
                fb = cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR)
                lms.append(detect_lm68(fb, det, align))
            seq = vp(video, lms)  # T x 96 x 96 灰度
            if seq is None: raise RuntimeError("no ROI")
            arr = np.array(seq).astype(np.float32) / 255.0
            arr = center_crop(arr, IMAGE_CROP)
            arr = (arr - IMAGE_MEAN)/IMAGE_STD
            arr = np.expand_dims(arr, -1)
            v = torch.from_numpy(arr.astype(np.float32)).permute(3,0,1,2).unsqueeze(0).cuda()
            T = v.shape[2]
            # 纯视觉：音频清零，长度对齐视频
            au = torch.zeros(1, 104, T).cuda()
            attn = torch.BoolTensor(au.size(0), au.size(-1)).fill_(False).cuda()
            with torch.no_grad():
                out = model.generate(au, attention_mask=attn, video=v, max_length=1024)
            hyp = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            hyp = ""
        rw,hw = ref.lower().split(), hyp.lower().split()
        raw_d += torchaudio.functional.edit_distance(rw,hw); raw_w += len(rw)
        nr,nh = normalize(ref).split(), normalize(hyp).split()
        norm_d += torchaudio.functional.edit_distance(nr,nh); norm_w += len(nr)
        results.append({"clip_id": r["clip_id"], "ref": ref, "hyp": hyp})
        if a.limit>0 or (i+1)%10==0:
            print(f"[{i+1}/{len(rows)}] raw={100*raw_d/raw_w:.2f}% norm={100*norm_d/norm_w:.2f}%")
            if a.limit>0:
                print(f"   REF: {ref}")
                print(f"   HYP: {hyp}")

    raw_wer, norm_wer = 100*raw_d/raw_w, 100*norm_d/norm_w
    print(f"\n{'='*40}\nAV-HuBERT BBC\nclips: {len(rows)}\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n{'='*40}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    with open(a.out.replace(".csv","_summary.txt"),"w") as f:
        f.write(f"Model: AV-HuBERT MuAViC-en\nDataset: BBC ({len(rows)})\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n")
    print(f"Saved: {a.out}")

if __name__ == "__main__":
    main()
