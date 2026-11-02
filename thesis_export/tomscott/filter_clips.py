"""
filter_clips.py  -  keep only clips where Tom Scott faces the camera (runs on CPU)

For each clip, 3 frames (20 / 50 / 80 %) are checked. A frame is OK when:
  1. the face is Tom      (SFace cosine similarity to a reference >= --sim)
  2. the face is big      (width >= --min-w pixels)
  3. the face is frontal  (nose close to the middle of the eyes, |yaw| < --max-yaw)
  4. only one person      (no other face >= 60 % of Tom's face width)
A clip is kept when at least 2 of its 3 frames are OK.

Models (OpenCV Zoo, small): put them in ./models/
  face_detection_yunet_2023mar.onnx
  face_recognition_sface_2021dec.onnx

Run in /root/autodl-tmp/tomscott:
  python filter_clips.py --videos 7DKv5H5Frt0 Zl_5LT2fzak     # quick check (2 videos)
  python filter_clips.py                                       # all videos
Outputs (in clips/):
  filter_result.csv     every clip + reason
  keep_manifest.csv     kept clips only  (used for ROI extraction and training)
  filter_by_video.csv   per-video summary
  _check/filter_kept.jpg, _check/filter_rejected.jpg   examples to look at
"""
import argparse
import os
import random
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="clips")
ap.add_argument("--models", default="models")
ap.add_argument("--ref-video", default="7DKv5H5Frt0", help="a video that is (almost) only Tom")
ap.add_argument("--videos", nargs="*", default=None, help="only these videos (for a quick test)")
ap.add_argument("--sim", type=float, default=0.40, help="SFace cosine threshold (OpenCV default 0.363)")
ap.add_argument("--min-w", type=int, default=60)
ap.add_argument("--max-yaw", type=float, default=0.30)
ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
args = ap.parse_args()

DET = os.path.join(args.models, "face_detection_yunet_2023mar.onnx")
REC = os.path.join(args.models, "face_recognition_sface_2021dec.onnx")
FRACS = (0.2, 0.5, 0.8)
G = {}


def init(ref=None):
    cv2.setNumThreads(1)
    G["det"] = cv2.FaceDetectorYN.create(DET, "", (320, 320), 0.8, 0.3, 5000)
    G["rec"] = cv2.FaceRecognizerSF.create(REC, "")
    G["ref"] = ref


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return []
    return [frames[min(int(len(frames) * p), len(frames) - 1)] for p in FRACS]


def detect(frame):
    """faces on a half-size image (fast), coordinates scaled back to full size"""
    h, w = frame.shape[:2]
    s = 0.5 if w > 900 else 1.0
    small = cv2.resize(frame, (int(w * s), int(h * s))) if s != 1.0 else frame
    G["det"].setInputSize((small.shape[1], small.shape[0]))
    _, faces = G["det"].detect(small)
    if faces is None:
        return []
    faces = faces.copy()
    faces[:, :14] /= s
    return list(faces)


def embed(frame, face):
    crop = G["rec"].alignCrop(frame, face)
    v = G["rec"].feature(crop).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def yaw_of(face):
    rex, rey, lex, ley, nx = face[4], face[5], face[6], face[7], face[8]
    eye_d = abs(lex - rex) + 1e-6
    return (nx - (rex + lex) / 2) / eye_d


def check_frame(frame):
    faces = [f for f in detect(frame) if f[2] >= 30]
    if not faces:
        return "no_face", 0.0, 0.0, 0.0
    sims = [float(embed(frame, f) @ G["ref"]) for f in faces]
    i = int(np.argmax(sims))
    tom, sim = faces[i], sims[i]
    w, yaw = float(tom[2]), float(yaw_of(tom))
    if sim < args.sim:
        return "not_tom", sim, w, yaw
    if w < args.min_w:
        return "small_face", sim, w, yaw
    if abs(yaw) >= args.max_yaw:
        return "side_face", sim, w, yaw
    if any(f[2] >= 0.6 * w for j, f in enumerate(faces) if j != i):
        return "two_people", sim, w, yaw
    return "ok", sim, w, yaw


def check_clip(row):
    try:
        frames = read_frames(os.path.join(args.clips, row["path"]))
    except Exception:
        frames = []
    if not frames:
        return {**row, "keep": False, "reason": "unreadable", "n_ok": 0, "sim": 0, "face_w": 0, "yaw": 0}
    res = [check_frame(f) for f in frames]
    n_ok = sum(r[0] == "ok" for r in res)
    need = 2 if len(res) >= 3 else len(res)
    keep = n_ok >= need
    bad = [r[0] for r in res if r[0] != "ok"]
    reason = "ok" if keep else Counter(bad).most_common(1)[0][0]
    return {**row, "keep": keep, "reason": reason, "n_ok": n_ok,
            "sim": round(float(np.median([r[1] for r in res])), 3),
            "face_w": round(float(np.median([r[2] for r in res])), 1),
            "yaw": round(float(np.median([r[3] for r in res])), 3)}


def ref_from_video(rows):
    """mean face embedding of the largest face in the reference video"""
    init(None)
    vecs = []
    for r in rows[:60]:
        fr = read_frames(os.path.join(args.clips, r["path"]))
        if not fr:
            continue
        faces = detect(fr[1])
        if faces:
            f = max(faces, key=lambda x: x[2])
            if f[2] >= args.min_w:
                vecs.append(embed(fr[1], f))
    vecs = np.array(vecs)
    ref = vecs.mean(0)
    ref /= np.linalg.norm(ref)
    sims = vecs @ ref
    vecs = vecs[sims >= 0.5]                      # drop odd ones (other people)
    ref = vecs.mean(0)
    ref /= np.linalg.norm(ref)
    print(f"reference built from {len(vecs)} faces of {args.ref_video}; "
          f"similarity to reference: median {np.median(vecs @ ref):.2f}, min {np.min(vecs @ ref):.2f}")
    return ref


def example_sheet(df, path, title_col):
    from PIL import Image, ImageDraw
    cols, rows_n, W, H = 6, 4, 240, 135
    sheet = Image.new("RGB", (cols * W, rows_n * (H + 15)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (_, r) in enumerate(df.head(cols * rows_n).iterrows()):
        fr = read_frames(os.path.join(args.clips, r["path"]))
        if not fr:
            continue
        img = Image.fromarray(cv2.cvtColor(fr[1], cv2.COLOR_BGR2RGB)).resize((W, H))
        x, y = (k % cols) * W, (k // cols) * (H + 15)
        sheet.paste(img, (x, y))
        d.rectangle([x, y + H, x + W, y + H + 15], fill="black")
        d.text((x + 3, y + H + 1), f"{r[title_col]}  sim {r['sim']:.2f}", fill="yellow")
    sheet.save(path, quality=85)


def main():
    for p in (DET, REC):
        if not os.path.exists(p):
            raise SystemExit(f"missing model: {p}")
    # read each finished video's own manifest (works while clipping is still running)
    import glob
    parts = [pd.read_csv(f, keep_default_na=False)
             for f in sorted(glob.glob(os.path.join(args.clips, "*", "manifest.csv")))]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise SystemExit(f"no clips/*/manifest.csv found in {args.clips}")
    man = pd.concat(parts, ignore_index=True)
    ex_file = os.path.join(args.clips, "exclude.txt")
    if os.path.exists(ex_file):
        excl = {l.strip() for l in open(ex_file, encoding="utf-8") if l.strip()}
        man = man[~man["video_id"].isin(excl)]
    print(f"{man['video_id'].nunique()} videos, {len(man)} clips found")
    if args.videos:
        man = man[man["video_id"].isin(args.videos + [args.ref_video])]
    ref_rows = man[man["video_id"] == args.ref_video].to_dict("records")
    if not ref_rows:
        raise SystemExit(f"reference video {args.ref_video} not found (is it clipped yet?)")
    ref = ref_from_video(ref_rows)

    rows = man.to_dict("records")
    if args.videos:
        rows = [r for r in rows if r["video_id"] in args.videos]
    print(f"checking {len(rows)} clips with {args.workers} workers ...")
    from tqdm import tqdm
    with Pool(args.workers, initializer=init, initargs=(ref,)) as pool:
        out = list(tqdm(pool.imap(check_clip, rows, chunksize=4), total=len(rows), ncols=90))

    df = pd.DataFrame(out)
    df["dur"] = df["dur"].astype(float)
    tag = "_test" if args.videos else ""
    df.to_csv(os.path.join(args.clips, f"filter_result{tag}.csv"), index=False)
    df[df["keep"]].drop(columns=["keep", "reason"]).to_csv(
        os.path.join(args.clips, f"keep_manifest{tag}.csv"), index=False)

    per = df.groupby("video_id").apply(lambda s: pd.Series({
        "clips": len(s), "kept": int(s["keep"].sum()), "keep_%": 100 * s["keep"].mean(),
        "kept_min": s.loc[s["keep"], "dur"].sum() / 60,
        "main_reject": Counter(s.loc[~s["keep"], "reason"]).most_common(1)[0][0] if (~s["keep"]).any() else "-"}))
    per = per.sort_values("keep_%", ascending=False)
    per.to_csv(os.path.join(args.clips, f"filter_by_video{tag}.csv"))
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 300)
    print("\nPER VIDEO")
    print(per.to_string(float_format=lambda x: f"{x:.1f}"))
    print("\nREASONS (all clips)")
    for k, v in df["reason"].value_counts().items():
        print(f"  {k:12s} {v:6d}  ({100 * v / len(df):.1f}%)")
    print(f"\nTOTAL kept {int(df['keep'].sum())} / {len(df)} clips, "
          f"{df.loc[df['keep'], 'dur'].sum() / 60:.1f} min of {df['dur'].sum() / 60:.1f} min")

    os.makedirs(os.path.join(args.clips, "_check"), exist_ok=True)
    random.seed(0)
    init(ref)
    example_sheet(df[df["keep"]].sample(min(24, int(df["keep"].sum())), random_state=0),
                  os.path.join(args.clips, "_check", f"filter_kept{tag}.jpg"), "video_id")
    rej = df[~df["keep"]]
    if len(rej):
        example_sheet(rej.sample(min(24, len(rej)), random_state=0),
                      os.path.join(args.clips, "_check", f"filter_rejected{tag}.jpg"), "reason")
    print(f"example sheets saved in {args.clips}/_check/")


if __name__ == "__main__":
    main()
