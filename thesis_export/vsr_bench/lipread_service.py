# -*- coding: utf-8 -*-
"""
唇语识别服务 —— 多模态感知引擎中的 "Lip Reading" 模块

输入：一段人脸视频（mp4 / webm / mov，建议 ≤ 10 秒）
处理：人脸检测 (RetinaFace) -> 关键点 (FAN) -> 嘴部对齐裁剪 96x96 -> VSR 模型
输出：识别文本 + 各阶段耗时 + 模型信息（JSON）

接口（其他感知模块按同样格式调用）：
  GET  /health                 服务状态、当前模型、设备
  POST /recognize              同步识别：form-data  file=<视频>   -> 结果 JSON
  POST /submit                 异步提交：form-data  file=<视频>   -> {"job_id": ...}
  GET  /status/{job_id}        查询异步结果 -> {"status": queued|running|done|error, ...}

返回的结果 JSON：
  {"text": "...", "model": "T1", "model_name": "...", "frames": 125, "video_seconds": 5.0,
   "timing": {"decode": 0.2, "face": 1.8, "infer": 0.7, "total": 2.7},
   "face_found_ratio": 0.98}

启动（每类模型一个进程，可以同时开多个，端口不同）：
  python lipread_service.py --model T1  --port 6006     # 监督
  python lipread_service.py --model T3  --port 6007     # 自监督（30 小时标签）
  python lipread_service.py --model U4g --port 6008     # USR 2.0 Huge 快速
  可选模型：T1 T3 B4 U4 U4g（同一类的模型不能在同一进程里，但不同进程可以同时运行）
"""
import argparse
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from common import ROOT

MODELS = {
    "T1": {"name": "Auto-AVSR (supervised, 3291 h labels)", "family": "autoavsr",
           "ckpt": f"{ROOT}/checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth"},
    "T3": {"name": "BRAVEn (self-supervised, 30 h labels)", "family": "raven",
           "ckpt": f"{ROOT}/checkpoints/raven/vsr_prelrs3vox2avs_large_ftlrs3trainvalvox2avs_selftrain_braven.pth"},
    "B4": {"name": "BRAVEn (self-supervised, 433 h labels)", "family": "raven",
           "ckpt": f"{ROOT}/checkpoints/new/B4_braven_high_large_avs_st.pth"},
    "U4": {"name": "USR 2.0 Huge (self-supervised, beam 40)", "family": "usr2", "beam": 40, "ctc": 0.1,
           "ckpt": f"{ROOT}/checkpoints/new/U4_usr2_high_huge.pth"},
    "U4g": {"name": "USR 2.0 Huge (self-supervised, fast decoding)", "family": "usr2", "beam": 1, "ctc": 0.0,
            "ckpt": f"{ROOT}/checkpoints/new/U4_usr2_high_huge.pth"},
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SECONDS = 20


# ======================= 视频读取 =======================
def to_mp4_25fps(path):
    """浏览器摄像头录的是 webm（时间戳常不规整），先用 ffmpeg 统一转成 25 fps 的 mp4"""
    import subprocess
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return path                                   # 没有 ffmpeg 就直接读
    out = path + ".25fps.mp4"
    r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", path, "-an", "-r", "25",
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(out):
        print("ffmpeg 转换失败，直接读取原文件:", r.stderr[-300:])
        return path
    return out


def read_video_25fps(path):
    """读任意格式视频，统一到 25 fps，返回 RGB (T, H, W, 3) uint8"""
    import av
    frames, times = [], []
    with av.open(path) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            times.append(float(f.time) if f.time is not None else len(times) / 25.0)
    if not frames:
        raise ValueError("视频里没有画面")
    dur = max(times[-1] - times[0], 1e-6) + 1 / 25.0
    n = max(1, int(round(dur * 25)))
    idx = np.searchsorted(np.array(times) - times[0], np.arange(n) / 25.0, side="right") - 1
    idx = np.clip(idx, 0, len(frames) - 1)
    return np.stack([frames[i] for i in idx]), dur


# ======================= 嘴部裁剪（和评测用的完全一样）=======================
class MouthCropper:
    def __init__(self):
        from cache_bbc_rois import get_68, load_98_to_68, load_video_process   # 统一流程，含 numpy 补丁
        from facexlib.alignment import init_alignment_model
        from facexlib.detection import init_detection_model
        self.get_68 = get_68
        self.to68 = load_98_to_68()
        self.vp = load_video_process()
        self.det = init_detection_model("retinaface_resnet50", half=False, device=DEVICE)
        self.align = init_alignment_model("awing_fan", device=DEVICE)

    def __call__(self, video):
        h, w = video.shape[1:3]
        scale = min(1.0, 640.0 / max(h, w))           # 大画面先缩小再检测，快很多
        lms, found = [], 0
        for frame in video:
            small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            with torch.no_grad():
                bb = self.det.detect_faces(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), 0.5)
            if bb is None or len(bb) == 0:
                lms.append(None)
                continue
            box = np.array(bb[0][:4], dtype=np.float32) / scale
            lms.append(self.get_68(frame, box, self.align, self.to68))
            found += 1
        seq = self.vp(video, lms)
        if seq is None:
            raise ValueError("没有检测到人脸，请正对摄像头、保证光线")
        seq = np.array(seq)
        if seq.ndim == 4:
            seq = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq])
        return seq.astype(np.uint8), found / len(video)


# ======================= 两个模型（同一进程只能加载一个）=======================
class AutoAVSR:
    def __init__(self, ckpt):
        from argparse import Namespace
        from eval_autoavsr_cache import load_model
        sys.path.insert(0, f"{ROOT}/code/auto_avsr")
        from datamodule.transforms import VideoTransform
        self.model = load_model(Namespace(ckpt=ckpt, convert_fn=""), DEVICE)
        self.vt = VideoTransform(subset="test")

    @torch.no_grad()
    def __call__(self, roi):
        vid = self.vt(torch.from_numpy(roi).unsqueeze(1).float()).to(DEVICE)
        return str(self.model(vid)).strip()


class USR2:
    def __init__(self, ckpt, beam, ctc):
        from eval_usr2 import build_beam_search, load_cfg, load_model, prep
        from espnet.asr.asr_utils import parse_hypothesis
        from utils.utils import UNIGRAM1000_LIST
        self.cfg = load_cfg("resnet_transformer_huge")
        self.model = load_model(self.cfg, ckpt, torch.device(DEVICE))
        self.bs = build_beam_search(self.model, beam, ctc, float(self.cfg.decode.penalty))
        self.bs.to(DEVICE)
        self.prep, self.parse, self.tokens = prep, parse_hypothesis, UNIGRAM1000_LIST

    @torch.no_grad()
    def __call__(self, roi):
        feat = self.model.encoder(xs_v=self.prep(roi).to(DEVICE))
        hyps = self.bs(x=feat.squeeze(0), modality="v",
                       maxlenratio=self.cfg.decode.maxlenratio, minlenratio=self.cfg.decode.minlenratio)
        text, _, _, _ = self.parse(hyps[0].asdict(), self.tokens)
        return text.replace("<eos>", "").replace("\u2581", " ").strip()


class Raven:
    """RAVEn / BRAVEn，用打过补丁的 GPU beam search（和评测完全一样）"""
    def __init__(self, ckpt):
        from argparse import Namespace
        import eval_raven_family as E
        self.cfg, learner, _ = E.build(Namespace(ckpt=ckpt, size="large", lm="", lm_weight=None, beam=40))
        learner = learner.to(DEVICE)
        learner.beam_search = learner.get_beam_search(learner.model)
        learner.beam_search.to(DEVICE)
        self.learner = learner.eval()
        from utils import UNIGRAM1000_LIST
        self.tokens, self.prep, self.to_text = UNIGRAM1000_LIST, E.prep, E.to_text

    @torch.no_grad()
    def __call__(self, roi):
        feat, _ = self.learner.model.encoder(self.prep(roi).to(DEVICE), None)
        nbest = self.learner.beam_search(x=feat.squeeze(0), maxlenratio=self.cfg.decode.maxlenratio,
                                         minlenratio=self.cfg.decode.minlenratio)
        return self.to_text(nbest, self.tokens)


def load_engine(mid):
    m = MODELS[mid]
    if not os.path.exists(m["ckpt"]):
        raise SystemExit(f"权重不存在: {m['ckpt']}")
    if m["family"] == "autoavsr":
        return AutoAVSR(m["ckpt"])
    if m["family"] == "raven":
        return Raven(m["ckpt"])
    return USR2(m["ckpt"], m["beam"], m["ctc"])


# ======================= 识别流程 =======================
ENGINE = {}


def recognize_file(path):
    t0 = time.perf_counter()
    conv = to_mp4_25fps(path)
    try:
        video, dur = read_video_25fps(conv)
    except Exception as e:
        raise ValueError(f"无法读取视频（{type(e).__name__}: {e}）")
    finally:
        if conv != path and os.path.exists(conv):
            os.remove(conv)
    if dur > MAX_SECONDS:
        raise ValueError(f"视频太长（{dur:.1f} 秒），请控制在 {MAX_SECONDS} 秒以内")
    t1 = time.perf_counter()
    roi, ratio = ENGINE["crop"](video)
    t2 = time.perf_counter()
    text = ENGINE["model"](roi)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    return {"text": text, "model": ENGINE["id"], "model_name": MODELS[ENGINE["id"]]["name"],
            "frames": int(len(roi)), "video_seconds": round(dur, 2), "face_found_ratio": round(ratio, 2),
            "timing": {"decode": round(t1 - t0, 2), "face": round(t2 - t1, 2),
                       "infer": round(t3 - t2, 2), "total": round(t3 - t0, 2)}}


# ======================= Web 服务 =======================
def build_app():
    from fastapi import FastAPI, File, HTTPException, UploadFile
    app = FastAPI(title="Lip Reading Module", version="1.0")
    jobs = {}
    pool = ThreadPoolExecutor(max_workers=1)          # 一块 GPU，排队处理
    lock = threading.Lock()

    def save(upload):
        suffix = os.path.splitext(upload.filename or "")[1] or ".mp4"
        fd, p = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(upload.file.read())
        return p

    def run(job_id, path):
        jobs[job_id]["status"] = "running"
        try:
            with lock:
                jobs[job_id].update(status="done", **recognize_file(path))
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))
            traceback.print_exc()
        finally:
            os.remove(path)

    @app.get("/health")
    def health():
        return {"ok": True, "model": ENGINE["id"], "model_name": MODELS[ENGINE["id"]]["name"],
                "device": DEVICE, "queued": sum(j["status"] in ("queued", "running") for j in jobs.values())}

    @app.post("/recognize")
    def recognize(file: UploadFile = File(...)):
        p = save(file)
        try:
            with lock:
                return recognize_file(p)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        finally:
            os.remove(p)

    @app.post("/submit")
    def submit(file: UploadFile = File(...)):
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"status": "queued", "submitted": time.time()}
        pool.submit(run, job_id, save(file))
        return {"job_id": job_id}

    @app.get("/status/{job_id}")
    def status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "unknown job")
        return {"job_id": job_id, **jobs[job_id]}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="T1")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--test", default="", help="只识别这个视频然后退出（不启动服务）")
    a = ap.parse_args()

    print(f"加载模型 {a.model}: {MODELS[a.model]['name']} ...")
    ENGINE["id"] = a.model
    ENGINE["crop"] = MouthCropper()
    ENGINE["model"] = load_engine(a.model)
    print("加载完成")

    if a.test:
        import json
        print(json.dumps(recognize_file(a.test), indent=2, ensure_ascii=False))
        raise SystemExit
    import uvicorn
    uvicorn.run(build_app(), host="0.0.0.0", port=a.port)
