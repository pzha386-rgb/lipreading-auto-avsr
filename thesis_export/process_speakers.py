"""
process_speakers.py - Multi-speaker SRT->clips->face-filter pipeline (YuNet only)
"""
import os, re, csv, sys, subprocess, glob
from pathlib import Path
from multiprocessing import Pool
import cv2
import numpy as np

INPUT_DIR = "/root/autodl-tmp/vsr_speakers"
OUTPUT_DIR = "/root/autodl-tmp/speaker_clips"
MIN_DUR, MAX_DUR, HARD_MAX = 3.0, 5.0, 6.0
SAMPLE_FRACS = (0.2, 0.5, 0.8)
MIN_FACE_WIDTH = 50
MAX_YAW = 0.35
FACE_AREA_RATIO = 0.02
KEEP_THRESHOLD = 2
NUM_WORKERS = min(8, max(2, (os.cpu_count() or 4) - 2))
YUNET_PATH = "/root/autodl-tmp/tomscott/models/face_detection_yunet_2023mar.onnx"

def time_to_seconds(t):
    h, m, rest = t.split(':')
    s, ms = rest.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def parse_srt(path):
    with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        content = f.read()
    blocks = re.split(r'\n\s*\n', content.strip())
    entries = []
    for block in blocks:
        lines = [l for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 2: continue
        time_idx = None
        for i, line in enumerate(lines):
            if '-->' in line: time_idx = i; break
        if time_idx is None: continue
        try:
            start_str, end_str = lines[time_idx].split('-->')
            start = time_to_seconds(start_str.strip())
            end = time_to_seconds(end_str.strip())
        except (ValueError, IndexError): continue
        text = ' '.join(lines[time_idx+1:]).strip()
        if text: entries.append({'start': start, 'end': end, 'text': text})
    return entries

def merge_clips(entries):
    clips = []
    i, n = 0, len(entries)
    while i < n:
        start, end = entries[i]['start'], entries[i]['end']
        text_parts = [entries[i]['text']]
        j = i + 1
        while j < n:
            cur_dur = end - start
            new_dur = entries[j]['end'] - start
            if cur_dur >= MIN_DUR:
                if new_dur > MAX_DUR: break
            else:
                if new_dur > HARD_MAX: break
            end = entries[j]['end']
            text_parts.append(entries[j]['text'])
            j += 1
            if (end - start) >= MAX_DUR: break
        text = ' '.join(text_parts)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[[^\]]*\]', '', text).strip()
        text = re.sub(r'\([^)]*\)', '', text).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        if text: clips.append({'start': start, 'end': end, 'text': text, 'dur': end-start})
        i = j
    if len(clips) >= 2 and clips[-1]['dur'] < MIN_DUR:
        merged = clips[-1]['end'] - clips[-2]['start']
        if merged <= HARD_MAX + 1.5:
            tail = clips.pop()
            clips[-1]['end'] = tail['end']
            clips[-1]['text'] += ' ' + tail['text']
            clips[-1]['dur'] = clips[-1]['end'] - clips[-1]['start']
    return clips

def cut_clip(input_video, start, end, output_path):
    cmd = ['ffmpeg','-y','-loglevel','error','-ss',f'{start:.3f}','-i',input_video,
           '-t',f'{end-start:.3f}','-r','25','-an','-c:v','libx264','-crf','18',
           '-pix_fmt','yuv420p',str(output_path)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0

# Face filter globals
_yunet = None

def init_worker():
    global _yunet
    cv2.setNumThreads(1)
    _yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.7, 0.3, 5000)

def check_frame_quality(frame):
    h, w = frame.shape[:2]
    frame_area = h * w
    s = 0.5 if w > 640 else 1.0
    small = cv2.resize(frame, (int(w*s), int(h*s))) if s != 1.0 else frame
    _yunet.setInputSize((small.shape[1], small.shape[0]))
    _, faces = _yunet.detect(small)
    if faces is None or len(faces) == 0: return "no_face"
    if len(faces) > 3: return "multi_face"
    faces[:, :14] /= s
    best = max(faces, key=lambda f: f[2] * f[3])
    face_w, face_area = float(best[2]), float(best[2]) * float(best[3])
    if face_area / frame_area < FACE_AREA_RATIO: return "small_face"
    if face_w < MIN_FACE_WIDTH: return "small_face"
    if best.shape[0] >= 10:
        rex, lex, nx = float(best[4]), float(best[6]), float(best[8])
        eye_d = abs(lex - rex) + 1e-6
        if abs((nx - (rex + lex) / 2) / eye_d) >= MAX_YAW: return "side_face"
    return "ok"

def check_clip_task(task):
    clip_path, info = task
    try:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened(): return {**info, "keep": False, "reason": "unreadable", "n_ok": 0}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 3:
            cap.release()
            return {**info, "keep": False, "reason": "too_short", "n_ok": 0}
        results = []
        for frac in SAMPLE_FRACS:
            idx = min(int(total * frac), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            results.append(check_frame_quality(frame) if ok else "unreadable")
        cap.release()
        n_ok = sum(1 for r in results if r == "ok")
        keep = n_ok >= KEEP_THRESHOLD
        if keep:
            reason = "ok"
        else:
            from collections import Counter
            bad = [r for r in results if r != "ok"]
            reason = Counter(bad).most_common(1)[0][0] if bad else "unknown"
        return {**info, "keep": keep, "reason": reason, "n_ok": n_ok}
    except Exception as e:
        return {**info, "keep": False, "reason": f"error", "n_ok": 0}

def process_video(video_path, srt_path, speaker_id, video_id, split, clip_out_dir):
    os.makedirs(clip_out_dir, exist_ok=True)
    manifest_file = os.path.join(clip_out_dir, "manifest.csv")
    if os.path.exists(manifest_file):
        rows = []
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f): rows.append(r)
        return rows
    entries = parse_srt(srt_path)
    if not entries: return []
    clips = merge_clips(entries)
    if not clips: return []
    rows = []
    for idx, clip in enumerate(clips):
        clip_id = f"clip_{idx:04d}"
        out_path = os.path.join(clip_out_dir, f"{clip_id}.mp4")
        if cut_clip(video_path, clip['start'], clip['end'], out_path):
            rows.append({'clip_id': clip_id, 'video_id': video_id, 'speaker_id': speaker_id,
                         'split': split, 'path': os.path.join(split, speaker_id, video_id, f"{clip_id}.mp4"),
                         'start': f"{clip['start']:.3f}", 'end': f"{clip['end']:.3f}",
                         'dur': f"{clip['dur']:.3f}", 'text': clip['text']})
    if rows:
        with open(manifest_file, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return rows

def main():
    if not os.path.exists(YUNET_PATH):
        print(f"ERROR: YuNet model not found at {YUNET_PATH}")
        return
    print("="*60); print("Multi-Speaker Processing Pipeline"); print("="*60)
    print(f"Input:  {INPUT_DIR}\nOutput: {OUTPUT_DIR}\n")
    os.makedirs(os.path.join(OUTPUT_DIR, "manifests"), exist_ok=True)

    print("STEP 1: Extracting clips from SRT...")
    all_rows = []
    speaker_stats = {}
    for split in ["train", "test"]:
        split_dir = os.path.join(INPUT_DIR, split)
        if not os.path.exists(split_dir): continue
        for speaker_id in sorted(os.listdir(split_dir)):
            spk_dir = os.path.join(split_dir, speaker_id)
            if not os.path.isdir(spk_dir): continue
            mp4s = sorted([f for f in os.listdir(spk_dir) if f.endswith('.mp4')])
            if not mp4s: continue
            print(f"\n  {split}/{speaker_id}: {len(mp4s)} videos")
            spk_clips, spk_dur = 0, 0.0
            for mp4 in mp4s:
                video_id = mp4.replace('.mp4', '')
                video_path = os.path.join(spk_dir, mp4)
                srt_path = None
                for f in os.listdir(spk_dir):
                    if video_id in f and f.endswith('.srt'):
                        srt_path = os.path.join(spk_dir, f); break
                if not srt_path:
                    print(f"    {video_id}: no SRT, skipping"); continue
                clip_out_dir = os.path.join(OUTPUT_DIR, split, speaker_id, video_id)
                rows = process_video(video_path, srt_path, speaker_id, video_id, split, clip_out_dir)
                all_rows.extend(rows)
                n, d = len(rows), sum(float(r['dur']) for r in rows)
                spk_clips += n; spk_dur += d
                print(f"    {video_id}: {n} clips, {d/60:.1f} min")
            speaker_stats[(split, speaker_id)] = {'clips': spk_clips, 'dur_min': spk_dur/60}
            print(f"    TOTAL: {spk_clips} clips, {spk_dur/60:.1f} min")
    print(f"\nSTEP 1 DONE: {len(all_rows)} clips total")

    print(f"\nSTEP 2: Face filtering ({NUM_WORKERS} workers)...")
    tasks = [(os.path.join(OUTPUT_DIR, r['path']), r) for r in all_rows]
    from tqdm import tqdm
    with Pool(NUM_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(check_clip_task, tasks, chunksize=8), total=len(tasks), ncols=80, desc="  Filter"))

    print(f"\nSTEP 3: Saving manifests...")
    kept = [r for r in results if r['keep']]
    rejected = [r for r in results if not r['keep']]
    print(f"  Kept: {len(kept)} / {len(results)}")
    from collections import Counter
    for reason, count in Counter(r['reason'] for r in rejected).most_common():
        print(f"    {reason}: {count}")

    manifest_dir = os.path.join(OUTPUT_DIR, "manifests")
    fields = ['clip_id','video_id','speaker_id','split','path','start','end','dur','text']
    train_all, test_all = [], []
    for (split, sid), _ in sorted(speaker_stats.items()):
        spk_kept = [r for r in kept if r['speaker_id']==sid and r['split']==split]
        if not spk_kept: continue
        with open(os.path.join(manifest_dir, f"{split}_{sid}.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(spk_kept)
        (train_all if split=="train" else test_all).extend(spk_kept)
    for name, data in [("train_all", train_all), ("test_all", test_all)]:
        with open(os.path.join(manifest_dir, f"{name}.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(data)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"\n{'Speaker':<22} {'Split':<7} {'Raw':<6} {'Kept':<6} {'Keep%':<7} {'Minutes'}")
    print("-"*65)
    for (split, sid), _ in sorted(speaker_stats.items()):
        sk = [r for r in kept if r['speaker_id']==sid and r['split']==split]
        sa = [r for r in results if r['speaker_id']==sid and r['split']==split]
        pct = 100*len(sk)/len(sa) if sa else 0
        mins = sum(float(r['dur']) for r in sk)/60
        print(f"  {sid:<20} {split:<7} {len(sa):<6} {len(sk):<6} {pct:<7.1f} {mins:.1f}")
    tmin = sum(float(r['dur']) for r in train_all)/60
    temin = sum(float(r['dur']) for r in test_all)/60
    print("-"*65)
    print(f"  TRAIN: {len(train_all)} clips, {tmin:.1f} min")
    print(f"  TEST:  {len(test_all)} clips, {temin:.1f} min")
    print(f"  TOTAL: {len(kept)} clips, {tmin+temin:.1f} min")

if __name__ == "__main__":
    main()
