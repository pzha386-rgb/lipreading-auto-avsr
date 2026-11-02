"""
make_clips_all.py  v2  -  cut ALL Tom Scott videos into 3-5 s clips (run LOCALLY)

v2 fixes:
- ffmpeg never waits for keyboard input (-nostdin)  -> no more freezing
- every ffmpeg call has a time limit (120 s); a stuck clip is skipped, not the whole run
- a clip is written to a temp name first, so a half-written file is never kept
- live progress bar for every video
- videos listed in clips\\exclude.txt are skipped completely

Put this file in D:\\tomscott together with make_clips.py, then:
  python make_clips_all.py
Finished clips are kept; running it again continues where it stopped.
"""
import csv
import glob
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from make_clips import parse_srt, merge_clips   # your SRT parser + 3-5 s merging

SRC = Path("tomscott_adapt")
OUT = Path("clips")
CHECK = OUT / "_check"
WORKERS = 4
TIMEOUT = 120
GRID_COLS, GRID_ROWS = 6, 5


def run_ffmpeg(args):
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error"] + args,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def cut(video, start, end, out_path):
    """same settings as your make_clips.py: 25 fps, no audio, H.264 crf 18"""
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True
    tmp = out_path.with_name(out_path.stem + "_tmp.mp4")
    ok = run_ffmpeg(["-threads", "1", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
                     "-r", "25", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                     "-threads", "2", str(tmp)])
    if ok and tmp.exists() and tmp.stat().st_size > 10_000:
        os.replace(tmp, out_path)
        return True
    if tmp.exists():
        tmp.unlink()
    return False


def extra_clean(text):
    t = text.replace("\u266a", " ").replace("&gt;", " ").replace(">>", " ")
    t = re.sub(r"(^|\s)-\s+", " ", t)
    t = re.sub(r"\b[A-Z][A-Z ]{1,20}:\s*", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def one_video(vid, srt, k, n):
    out_dir = OUT / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*_tmp.mp4"):
        f.unlink()
    clips = merge_clips(parse_srt(srt), min_dur=3.0, max_dur=5.0)
    jobs = []
    for idx, c in enumerate(clips):
        c["text"] = extra_clean(c["text"])
        if c["text"]:
            jobs.append((f"clip_{idx:04d}", c))

    video = SRC / f"{vid}.mp4"
    ok_ids, failed = set(), []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(cut, video, c["start"], c["end"], out_dir / f"{cid}.mp4"): cid for cid, c in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"[{k}/{n}] {vid} cut  ", ncols=90):
            (ok_ids.add if fut.result() else failed.append)(futs[fut])

    rows = [{"video_id": vid, "clip_id": cid, "path": f"{vid}/{cid}.mp4",
             "start": f"{c['start']:.3f}", "end": f"{c['end']:.3f}", "dur": f"{c['dur']:.3f}",
             "text": c["text"]} for cid, c in jobs if cid in ok_ids]
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "clip_id", "path", "start", "end", "dur", "text"])
        w.writeheader()
        w.writerows(rows)
    if failed:
        print(f"    {len(failed)} clips failed (skipped): {failed[:5]}")
    return rows


def contact_sheets(vid, rows, k, n):
    from PIL import Image, ImageDraw
    CHECK.mkdir(parents=True, exist_ok=True)
    tmp = CHECK / "_tmp"
    tmp.mkdir(exist_ok=True)

    def thumb(r):
        jpg = tmp / f"{vid}_{r['clip_id']}.jpg"
        run_ffmpeg(["-threads", "1", "-ss", f"{float(r['dur']) / 2:.2f}", "-i", str(OUT / r["path"]),
                    "-frames:v", "1", "-vf", "scale=240:-2", str(jpg)])
        return r["clip_id"], jpg

    with ThreadPoolExecutor(WORKERS) as ex:
        thumbs = list(tqdm(ex.map(thumb, rows), total=len(rows), desc=f"[{k}/{n}] {vid} check", ncols=90))
    per = GRID_COLS * GRID_ROWS
    for page in range(0, len(thumbs), per):
        sheet = Image.new("RGB", (GRID_COLS * 240, GRID_ROWS * 150), "white")
        d = ImageDraw.Draw(sheet)
        for i, (cid, jpg) in enumerate(thumbs[page:page + per]):
            x, y = (i % GRID_COLS) * 240, (i // GRID_COLS) * 150
            if jpg.exists():
                sheet.paste(Image.open(jpg).resize((240, 135)), (x, y))
            d.rectangle([x, y + 135, x + 240, y + 150], fill="black")
            d.text((x + 4, y + 136), cid, fill="yellow")
        sheet.save(CHECK / f"{vid}_{page // per + 1:02d}.jpg", quality=85)
    for f in tmp.glob("*.jpg"):
        f.unlink()


def main():
    global WORKERS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2),
                    help="parallel ffmpeg jobs (default: CPU cores / 2; each job uses 2 threads)")
    ap.add_argument("--no-check", action="store_true", help="skip contact sheets (faster)")
    a = ap.parse_args()
    WORKERS = a.workers
    print(f"using {WORKERS} parallel ffmpeg jobs")
    excl = set()
    ex_file = OUT / "exclude.txt"
    if ex_file.exists():
        excl = {l.strip() for l in open(ex_file, encoding="utf-8") if l.strip() and "/" not in l}
    vids = [v for v in sorted(Path(p).stem for p in glob.glob(str(SRC / "*.mp4"))) if v not in excl]
    print(f"{len(vids)} videos to process (skipped by exclude.txt: {sorted(excl) or 'none'})\n")

    all_rows = []
    for k, vid in enumerate(vids, 1):
        srts = sorted(SRC.glob(f"{vid}.en*.srt"))
        if not srts:
            print(f"skip {vid}: no srt")
            continue
        rows = one_video(vid, srts[0], k, len(vids))
        sheet_done = (CHECK / f"{vid}_01.jpg").exists()
        if rows and not sheet_done and not a.no_check:
            contact_sheets(vid, rows, k, len(vids))
        mins = sum(float(r["dur"]) for r in rows) / 60
        print(f"    -> {vid}: {len(rows)} clips, {mins:.1f} min of speech\n")
        all_rows += rows

    with open(OUT / "all_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "clip_id", "path", "start", "end", "dur", "text"])
        w.writeheader()
        w.writerows(all_rows)
    total = sum(float(r["dur"]) for r in all_rows) / 60
    print(f"TOTAL: {len(vids)} videos, {len(all_rows)} clips, {total:.1f} min")
    print(f"next: check the contact sheets in {CHECK}")


if __name__ == "__main__":
    main()
