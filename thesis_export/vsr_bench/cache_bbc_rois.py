# -*- coding: utf-8 -*-
"""
BBC / 自录视频：人脸检测只做一次，保存嘴部 ROI，之后所有模型都读这个缓存
（以前每个模型都要重新检测一遍人脸，现在 30 个模型只检测 1 次；
  而且保证所有模型看到的是完全相同的嘴部画面）

流程和你之前的脚本一样：facexlib RetinaFace -> AWing-FAN 98 点 -> 68 点
                       -> Auto-AVSR 的 VideoProcess（仿射对齐 + 96x96 裁剪）-> 灰度

输出：/root/autodl-tmp/cache/{name}/{clip_id}.npy + manifest.csv

运行：
  # BBC（manifest 列名：clip_id, path, text —— 和你之前的一样）
  python cache_bbc_rois.py --name bbc --clips-dir <BBC视频文件夹> --manifest <manifest.csv>
  # 自录视频
  python cache_bbc_rois.py --name self --clips-dir <文件夹> --manifest <manifest.csv>
  # 先试 3 段
  python cache_bbc_rois.py --name bbc ... --limit 3
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
# facexlib 还在用 numpy 旧写法（np.float 等），新版 numpy 已删除，这里补回来
for _n, _t in [("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
import torch
import torchvision

from common import CACHE, CODE_AUTOAVSR


def load_video_process():
    """从 auto_avsr 仓库导入 VideoProcess（你之前用的就是这个）"""
    cands = glob.glob(f"{CODE_AUTOAVSR}/preparation/detectors/retinaface/video_process.py")
    if not cands:
        cands = glob.glob(f"{CODE_AUTOAVSR}/**/video_process.py", recursive=True)
    if not cands:
        raise SystemExit("找不到 auto_avsr 的 video_process.py")
    d = os.path.dirname(cands[0])
    sys.path.insert(0, d)
    from video_process import VideoProcess
    mean_face = os.path.join(d, "20words_mean_face.npy")
    print("VideoProcess:", cands[0])
    return VideoProcess(mean_face_path=mean_face, convert_gray=False)


def load_98_to_68():
    for mod in ["facexlib.alignment", "facexlib.utils.face_utils", "facexlib.utils"]:
        try:
            m = __import__(mod, fromlist=["landmark_98_to_68"])
            return m.landmark_98_to_68
        except (ImportError, AttributeError):
            continue
    raise SystemExit("facexlib 里找不到 landmark_98_to_68")


def get_68(frame_rgb, bbox, align_model, to68, margin=0.3):
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - bw * margin)), int(max(0, y1 - bh * margin))
    x2, y2 = int(min(w, x2 + bw * margin)), int(min(h, y2 + bh * margin))
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    with torch.no_grad():
        lm98 = align_model.get_landmarks(crop)
    lm98 = np.asarray(lm98, dtype=np.float32).reshape(-1, 2)
    lm98[:, 0] += x1
    lm98[:, 1] += y1
    return np.asarray(to68(lm98), dtype=np.float32)


def from_raw(a, rows, out_dir):
    from facexlib.alignment import init_alignment_model
    from facexlib.detection import init_detection_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = init_detection_model("retinaface_resnet50", half=False, device=device)
    align = init_alignment_model("awing_fan", device=device)
    to68 = load_98_to_68()
    vp = load_video_process()

    ok, bad = [], []
    for i, r in enumerate(rows):
        cid, npy = r["clip_id"], f"{r['clip_id']}.npy"
        if os.path.exists(os.path.join(out_dir, npy)):
            ok.append({"utt_id": cid, "text": r["text"].strip(), "npy": npy})
            continue
        try:
            video = torchvision.io.read_video(os.path.join(a.clips_dir, r["path"]), pts_unit="sec")[0].numpy()
            lms = []
            for t in range(video.shape[0]):
                with torch.no_grad():
                    bb = det.detect_faces(cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR), 0.5)
                lms.append(None if bb is None or len(bb) == 0 else get_68(video[t], bb[0], align, to68))
            seq = vp(video, lms)
            if seq is None:
                raise RuntimeError("no face found in the clip")
            seq = np.array(seq)
            if seq.ndim == 4:
                seq = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in seq])
            np.save(os.path.join(out_dir, npy), seq.astype(np.uint8))
            ok.append({"utt_id": cid, "text": r["text"].strip(), "npy": npy})
            print(f"[{i + 1}/{len(rows)}] {cid}  frames={len(seq)}  shape={seq.shape[1:]}")
        except Exception as e:
            bad.append(cid)
            print(f"[{i + 1}/{len(rows)}] {cid}  FAILED: {e}")
    return ok, bad


def from_roi_videos(a, rows, out_dir):
    """备用：如果你已经用 extract_rois.py 提取过 ROI 视频，直接转成 npy"""
    ok, bad = [], []
    for r in rows:
        cid = r["clip_id"]
        p = os.path.join(a.roi_videos, r["path"]) if os.path.exists(os.path.join(a.roi_videos, r["path"])) \
            else os.path.join(a.roi_videos, f"{cid}.mp4")
        try:
            v = torchvision.io.read_video(p, pts_unit="sec")[0].numpy()
            g = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in v]).astype(np.uint8)
            if g.shape[1:] != (96, 96):
                g = np.stack([cv2.resize(f, (96, 96)) for f in g])
            np.save(os.path.join(out_dir, f"{cid}.npy"), g)
            ok.append({"utt_id": cid, "text": r["text"].strip(), "npy": f"{cid}.npy"})
        except Exception as e:
            bad.append(cid)
            print(cid, "FAILED:", e)
    return ok, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="缓存名字：bbc / self / ...")
    ap.add_argument("--clips-dir", help="原始视频文件夹")
    ap.add_argument("--manifest", required=True, help="CSV，列：clip_id, path, text")
    ap.add_argument("--roi-videos", help="（备用）已经提取好的 ROI 视频文件夹")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    out_dir = os.path.join(CACHE, a.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(a.manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} clips")

    ok, bad = from_roi_videos(a, rows, out_dir) if a.roi_videos else from_raw(a, rows, out_dir)
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "text", "npy"])
        w.writeheader()
        w.writerows(ok)
    print(f"\n完成 {len(ok)} 段，失败 {len(bad)} 段 -> {out_dir}")
    if bad:
        print("失败的片段:", bad)


if __name__ == "__main__":
    main()
