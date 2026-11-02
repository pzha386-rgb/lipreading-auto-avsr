#!/usr/bin/env python
"""BRAVEn 在 BBC 上评估"""
import os, sys, argparse, re, csv
# raven 的 espnet 优先（auto_avsr 也有 espnet 会冲突）
sys.path.insert(0, "/root/autodl-tmp/code/raven")

import numpy as np
if not hasattr(np,'float'): np.float=float
if not hasattr(np,'int'): np.int=int
if not hasattr(np,'bool'): np.bool=bool
import cv2
import pandas as pd
import torch
import torchaudio
import torchvision
import inflect
from omegaconf import OmegaConf
import importlib.util
_vp_spec = importlib.util.spec_from_file_location(
    "video_process",
    "/root/autodl-tmp/code/auto_avsr/preparation/detectors/retinaface/video_process.py")
_vp_mod = importlib.util.module_from_spec(_vp_spec)
_vp_spec.loader.exec_module(_vp_mod)
VideoProcess = _vp_mod.VideoProcess
from facexlib.detection import init_detection_model
from facexlib.alignment import init_alignment_model, landmark_98_to_68

p = inflect.engine()
def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    def n2w(m):
        try: return p.number_to_words(m.group(),andword='').replace('-',' ').replace(',',' ')
        except: return m.group()
    text = re.sub(r'\d+', n2w, text)
    return re.sub(r'\s+',' ',text).strip()

CROP=88; MEAN=0.421; STD=0.165

def detect_lm68(frame_bgr, det, align):
    with torch.no_grad():
        bb = det.detect_faces(frame_bgr, 0.5)
    if bb is None or len(bb)==0: return None
    b = bb[0][:4].astype(int); x1,y1,x2,y2=b; h,w=frame_bgr.shape[:2]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
    face=frame_bgr[y1:y2,x1:x2]
    if face.size==0: return None
    with torch.no_grad():
        lm=align.get_landmarks(face)
    if lm is None or len(lm)==0: return None
    lm=np.array(lm)
    if lm.ndim==3: lm=lm[0]
    lm[:,0]+=x1; lm[:,1]+=y1
    lm68=landmark_98_to_68(lm)
    lm68=np.array(lm68)
    if lm68.ndim==3: lm68=lm68[0]
    return lm68

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--clips-dir",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    device="cuda"

    base="/root/autodl-tmp/code/raven/conf"
    cfg=OmegaConf.create({
        "data":{"modality":"video","labels_type":"unigram1000"},
        "model":OmegaConf.merge(
            OmegaConf.load(f"{base}/model/finetune.yaml"),
            {"visual_backbone":OmegaConf.load(f"{base}/model/visual_backbone/resnet_transformer_large.yaml"),
             "language_model":OmegaConf.load(f"{base}/model/language_model/default.yaml"),
             "pretrained_model_path":a.ckpt,"pretrained_lm_path":None}),
        "decode":OmegaConf.load(f"{base}/decode/default.yaml"),
    })
    cfg.model.visual_backbone.ddim=cfg.model.visual_backbone.adim
    cfg.model.visual_backbone.dheads=cfg.model.visual_backbone.aheads
    cfg.model.visual_backbone.dunits=cfg.model.visual_backbone.eunits

    from finetune_learner import Learner
    print("构建 BRAVEn...")
    learner=Learner(cfg).to('cpu').eval()
    learner.model.encoder=learner.model.encoder.to(device)
    print("就绪")

    det=init_detection_model('retinaface_resnet50',half=False,device=device)
    align=init_alignment_model('awing_fan',device=device)
    vp=VideoProcess(convert_gray=True)

    from espnet.asr.asr_utils import add_results_to_json
    from utils import UNIGRAM1000_LIST
    token_list=UNIGRAM1000_LIST

    rows=[]
    with open(a.manifest,encoding="utf-8") as f:
        for r in csv.DictReader(f): rows.append(r)
    if a.limit>0: rows=rows[:a.limit]
    print(f"clips: {len(rows)}")

    results,raw_d,raw_w,norm_d,norm_w=[],0,0,0,0
    for i,r in enumerate(rows):
        ref=r["text"].strip()
        try:
            video=torchvision.io.read_video(os.path.join(a.clips_dir,r["path"]),pts_unit="sec")[0].numpy()
            lms=[]
            for t in range(video.shape[0]):
                fb=cv2.cvtColor(video[t],cv2.COLOR_RGB2BGR)
                lms.append(detect_lm68(fb,det,align))
            seq=vp(video,lms)  # T x 96 x 96 灰度
            if seq is None: raise RuntimeError("no ROI")
            arr=np.array(seq).astype(np.float32)/255.0
            T,H,W=arr.shape
            ii,jj=(H-CROP)//2,(W-CROP)//2
            arr=arr[:,ii:ii+CROP,jj:jj+CROP]
            arr=(arr-MEAN)/STD
            v=torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(device)  # 1 x T x 88 x 88
            with torch.no_grad():
                feat,_=learner.model.encoder(v,None)
                feat=feat.squeeze(0).cpu()
                nbest=learner.beam_search(x=feat,maxlenratio=cfg.decode.maxlenratio,minlenratio=cfg.decode.minlenratio)
            nbest=[h.asdict() for h in nbest[:1]]
            hyp=add_results_to_json(nbest,token_list).replace("<eos>","").replace("▁"," ").strip()
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            hyp=""
        rw,hw=ref.lower().split(),hyp.lower().split()
        raw_d+=torchaudio.functional.edit_distance(rw,hw); raw_w+=len(rw)
        nr,nh=normalize(ref).split(),normalize(hyp).split()
        norm_d+=torchaudio.functional.edit_distance(nr,nh); norm_w+=len(nr)
        results.append({"clip_id":r["clip_id"],"ref":ref,"hyp":hyp})
        if a.limit>0 or (i+1)%10==0:
            print(f"[{i+1}/{len(rows)}] raw={100*raw_d/raw_w:.2f}% norm={100*norm_d/norm_w:.2f}%")
            if a.limit>0:
                print(f"   REF: {ref}")
                print(f"   HYP: {hyp}")

    raw_wer,norm_wer=100*raw_d/raw_w,100*norm_d/norm_w
    print(f"\n{'='*40}\nBRAVEn BBC\nclips: {len(rows)}\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n{'='*40}")
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    pd.DataFrame(results).to_csv(a.out,index=False)
    with open(a.out.replace(".csv","_summary.txt"),"w") as f:
        f.write(f"Model: BRAVEn Large\nDataset: BBC ({len(rows)})\nRaw WER: {raw_wer:.2f}%\nNorm WER: {norm_wer:.2f}%\n")
    print(f"Saved: {a.out}")

if __name__=="__main__":
    main()
