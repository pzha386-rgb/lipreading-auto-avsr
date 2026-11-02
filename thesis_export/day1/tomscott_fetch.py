"""
tomscott_fetch.py  -  run on your LOCAL computer (NZ network)
Downloads Tom Scott videos that have HUMAN English subtitles (.srt),
skipping the video already used for the BBC test set.

pip install -U yt-dlp      (ffmpeg must be on PATH for merging + srt)

Step 1 - list only (check first, no download):
python tomscott_fetch.py --exclude TEST_VIDEO_ID --list-only
Step 2 - download:
python tomscott_fetch.py --exclude TEST_VIDEO_ID --n 10
"""
import argparse
import json
import os
import yt_dlp

ap = argparse.ArgumentParser()
ap.add_argument("--channel", default="https://www.youtube.com/@TomScottGo/videos")
ap.add_argument("--exclude", nargs="*", default=[], help="YouTube id(s) of the BBC test video")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--scan", type=int, default=80, help="how many channel videos to check")
ap.add_argument("--min-dur", type=int, default=180)
ap.add_argument("--max-dur", type=int, default=900)
ap.add_argument("--out", default="tomscott_adapt")
ap.add_argument("--list-only", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True, "playlistend": args.scan}) as y:
    entries = y.extract_info(args.channel, download=False)["entries"]

picked = []
with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as y:
    for e in entries:
        vid = e["id"]
        if vid in args.exclude:
            print(f"skip (test video) {vid}")
            continue
        try:
            info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
        except Exception as ex:
            print(f"skip {vid}: {ex}")
            continue
        dur = info.get("duration") or 0
        langs = [l for l in (info.get("subtitles") or {}) if l.startswith("en")]  # human subs only
        ok = langs and args.min_dur <= dur <= args.max_dur
        print(f"{'OK ' if ok else '-- '} {vid}  {dur:4d}s  subs={langs}  {info.get('title', '')[:60]}")
        if ok:
            picked.append({"id": vid, "title": info.get("title"), "duration": dur, "sub_lang": langs[0]})
        if len(picked) >= args.n:
            break

json.dump(picked, open(f"{args.out}/manifest.json", "w"), indent=2, ensure_ascii=False)
total = sum(p["duration"] for p in picked) / 60
print(f"\npicked {len(picked)} videos, about {total:.0f} min total -> {args.out}/manifest.json")
if args.list_only:
    raise SystemExit

for p in picked:
    opts = {
        "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]",
        "merge_output_format": "mp4",
        "outtmpl": f"{args.out}/%(id)s.%(ext)s",
        "writesubtitles": True,
        "subtitleslangs": [p["sub_lang"]],
        "postprocessors": [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
    }
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([f"https://www.youtube.com/watch?v={p['id']}"])
print("done. next: cut clips with make_clips.py, then upload to /root/autodl-tmp/")
