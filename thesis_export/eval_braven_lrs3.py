#!/usr/bin/env python
"""BRAVEn 在 LRS3 上评估 - 复用 RAVEn Learner，绕过 SLURM/Hydra"""
import os, sys, argparse, re, glob
sys.path.insert(0, "/root/autodl-tmp/code/raven")

import numpy as np
import pandas as pd
import torch
import torchaudio
import inflect
from omegaconf import OmegaConf
from hydra import initialize, compose

p = inflect.engine()
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    def n2w(m):
        try: return p.number_to_words(m.group(), andword='').replace('-',' ').replace(',',' ')
        except: return m.group()
    text = re.sub(r'\d+', n2w, text)
    return re.sub(r'\s+', ' ', text).strip()

CROP=88; MEAN=0.421; STD=0.165

def prep_video(video_frames):
    frames = []
    for f in video_frames:
        rows = [np.asarray(r, dtype=np.uint8) for r in f]
        frames.append(np.stack(rows))
    arr = np.stack(frames).astype(np.float32)  # T x 96 x 96
    arr = arr / 255.0
    T,H,W = arr.shape
    i,j = (H-CROP)//2, (W-CROP)//2
    arr = arr[:, i:i+CROP, j:j+CROP]  # T x 88 x 88
    arr = (arr - MEAN)/STD
    # RAVEn video 输入: [1, T, 88, 88]（灰度单通道，squeeze 后）
    v = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)  # 1 x T x 88 x 88
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda"

    # 手动构造 cfg（绕过 Hydra 的 SLURM 依赖）
    with initialize(config_path="code/raven/conf", version_base=None):
        pass  # 用 compose 加载
    # 直接用 OmegaConf 加载各配置文件拼装
    base = "/root/autodl-tmp/code/raven/conf"
    cfg = OmegaConf.create({
        "data": {
            "modality": "video",
            "labels_type": "unigram1000",
        },
        "model": OmegaConf.merge(
            OmegaConf.load(f"{base}/model/finetune.yaml"),
            {"visual_backbone": OmegaConf.load(f"{base}/model/visual_backbone/resnet_transformer_large.yaml"),
             "language_model": OmegaConf.load(f"{base}/model/language_model/default.yaml"),
             "pretrained_model_path": a.ckpt,
             "pretrained_lm_path": None}
        ),
        "decode": OmegaConf.merge(OmegaConf.load(f"{base}/decode/default.yaml"), {"beam_size": 40}),
    })
    # 解析 large backbone 里的 ${...} 引用
    cfg.model.visual_backbone.ddim = cfg.model.visual_backbone.adim
    cfg.model.visual_backbone.dheads = cfg.model.visual_backbone.aheads
    cfg.model.visual_backbone.dunits = cfg.model.visual_backbone.eunits

    from finetune_learner import Learner
    print("构建 BRAVEn 模型...")
    learner = Learner(cfg)
    learner = learner.to('cpu').eval()
    learner.model.encoder = learner.model.encoder.to(device)  # encoder用GPU加速
    print("模型就绪")

    files = sorted(glob.glob(os.path.join(a.parquet_dir, "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if a.limit>0: df = df.head(a.limit)
    print(f"样本数: {len(df)}")

    from espnet.asr.asr_utils import add_results_to_json
    from utils import UNIGRAM1000_LIST
    token_list = UNIGRAM1000_LIST

    results, raw_d,raw_w,norm_d,norm_w = [],0,0,0,0
    for idx, row in df.iterrows():
        ref = str(row['label']).strip()
        try:
            v = prep_video(row['video']).to(device)  # 1 x T x 88 x 88
            with torch.no_grad():
                feat, _ = learner.model.encoder(v, None)
                feat = feat.squeeze(0).cpu()  # 转CPU给beam_search
                nbest = learner.beam_search(
                    x=feat,
                    maxlenratio=cfg.decode.maxlenratio,
                    minlenratio=cfg.decode.minlenratio)
            nbest = [h.asdict() for h in nbest[:1]]
            hyp = add_results_to_json(nbest, token_list).replace("<eos>","")
            hyp = hyp.replace("▁"," ").strip()
        except Exception as e:
            print(f"[{idx}] ERROR: {e}")
            hyp = ""
        rw,hw = ref.lower().split(), hyp.lower().split()
        raw_d += torchaudio.functional.edit_distance(rw,hw); raw_w += len(rw)
        nr,nh = normalize(ref).split(), normalize(hyp).split()
        norm_d += torchaudio.functional.edit_distance(nr,nh); norm_w += len(nr)
        results.append({"idx": idx, "ref": ref, "hyp": hyp})
        if a.limit>0:
            print(f"[{idx}] REF: {ref}")
            print(f"[{idx}] HYP: {hyp}")
            print("-"*30)
        if True:  # 每条都打印
            print(f"[{idx+1}/{len(df)}] raw={100*raw_d/raw_w:.2f}% norm={100*norm_d/norm_w:.2f}%")

    raw_wer, norm_wer = 100*raw_d/raw_w, 100*norm_d/norm_w
    print(f"\n{'='*40}\nBRAVEn LRS3\nSamples: {len(df)}\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n{'='*40}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(results).to_csv(a.out, index=False)
    with open(a.out.replace(".csv","_summary.txt"),"w") as f:
        f.write(f"Model: BRAVEn Large\nDataset: LRS3 test ({len(df)})\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n")
    print(f"Saved: {a.out}")

if __name__ == "__main__":
    main()
