#!/usr/bin/env python
"""AV-HuBERT (nguyenvulebinh MuAViC-en) 在 LRS3 上评估"""
import os, sys, argparse, re
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, "/root/autodl-tmp/code/AV-HuBERT-S2S")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from python_speech_features import logfbank
import inflect

from src.model.avhubert2text import AV2TextForConditionalGeneration
from transformers import Speech2TextTokenizer

p = inflect.engine()
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    def n2w(m):
        try: return p.number_to_words(m.group(), andword='').replace('-',' ').replace(',',' ')
        except: return m.group()
    text = re.sub(r'\d+', n2w, text)
    return re.sub(r'\s+', ' ', text).strip()

IMAGE_CROP = 88
IMAGE_MEAN = 0.421
IMAGE_STD = 0.165

def center_crop(frames, size):
    # frames: T x H x W
    T, H, W = frames.shape
    th = tw = size
    i = (H - th) // 2
    j = (W - tw) // 2
    return frames[:, i:i+th, j:j+tw]

def prep_video(video_frames):
    # 每帧 (96,) object -> 96x96
    frames = []
    for f in video_frames:
        rows = [np.asarray(r, dtype=np.uint8) for r in f]
        frames.append(np.stack(rows))
    arr = np.stack(frames).astype(np.float32)  # T x 96 x 96
    arr = arr / 255.0
    arr = center_crop(arr, IMAGE_CROP)  # T x 88 x 88
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = np.expand_dims(arr, axis=-1)  # T x 88 x 88 x 1
    v = torch.from_numpy(arr.astype(np.float32))
    v = v.permute(3, 0, 1, 2).unsqueeze(0)  # 1 x 1 x T x 88 x 88
    return v

def stacker(feats, stack_order=4):
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        feats = np.concatenate([feats, np.zeros([res, feat_dim]).astype(feats.dtype)], axis=0)
    return feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)

def prep_audio(audio, n_video_frames):
    wav = np.array(audio).astype(np.float32)
    feats = logfbank(wav, samplerate=16000).astype(np.float32)  # T x 26
    feats = stacker(feats, 4)  # T' x 104
    a = torch.from_numpy(feats.astype(np.float32))
    # 对齐到视频帧数
    diff = len(a) - n_video_frames
    if diff < 0:
        a = torch.cat([a, torch.zeros(-diff, a.shape[-1])])
    elif diff > 0:
        a = a[:-diff]
    with torch.no_grad():
        a = F.layer_norm(a, a.shape[1:])
    a = a.permute(1, 0).unsqueeze(0)  # 1 x 104 x T
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda"
    print("加载 AV-HuBERT...")
    model = AV2TextForConditionalGeneration.from_pretrained(
        "nguyenvulebinh/AV-HuBERT-MuAViC-en",
        cache_dir="/root/autodl-tmp/checkpoints/avhubert_s2s")
    tokenizer = Speech2TextTokenizer.from_pretrained(
        "nguyenvulebinh/AV-HuBERT-MuAViC-en",
        cache_dir="/root/autodl-tmp/checkpoints/avhubert_s2s")
    model = model.cuda().eval()
    print("模型就绪")

    import glob
    files = sorted(glob.glob(os.path.join(a.parquet_dir, "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if a.limit > 0:
        df = df.head(a.limit)
    print(f"样本数: {len(df)}")

    results = []
    raw_d, raw_w, norm_d, norm_w = 0,0,0,0
    for i, row in df.iterrows():
        ref = str(row['label']).strip()
        try:
            v = prep_video(row['video']).cuda()
            n_vf = v.shape[2]
            # 纯视觉模式：音频置零，只靠读唇
            au = prep_audio(row['audio'], n_vf).cuda()
            au = torch.zeros_like(au)  # 音频清零
            attn = torch.BoolTensor(au.size(0), au.size(-1)).fill_(False).cuda()
            with torch.no_grad():
                out = model.generate(au, attention_mask=attn, video=v, max_length=1024)
            hyp = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            hyp = ""
        # raw
        rw, hw = ref.lower().split(), hyp.lower().split()
        raw_d += torchaudio.functional.edit_distance(rw, hw); raw_w += len(rw)
        # norm
        nr, nh = normalize(ref).split(), normalize(hyp).split()
        norm_d += torchaudio.functional.edit_distance(nr, nh); norm_w += len(nr)
        results.append({"idx": i, "ref": ref, "hyp": hyp})
        if a.limit > 0:
            print(f"[{i}] REF: {ref}")
            print(f"[{i}] HYP: {hyp}")
            print("-"*30)
        if (i+1) % 50 == 0:
            print(f"[{i+1}/{len(df)}] raw={100*raw_d/raw_w:.2f}% norm={100*norm_d/norm_w:.2f}%")

    raw_wer = 100*raw_d/raw_w
    norm_wer = 100*norm_d/norm_w
    print(f"\n{'='*40}\nAV-HuBERT LRS3\nSamples: {len(df)}\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n{'='*40}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    with open(a.out.replace(".csv","_summary.txt"),"w") as f:
        f.write(f"Model: AV-HuBERT MuAViC-en\nDataset: LRS3 test ({len(df)})\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n")
    print(f"Saved: {a.out}")

if __name__ == "__main__":
    main()
