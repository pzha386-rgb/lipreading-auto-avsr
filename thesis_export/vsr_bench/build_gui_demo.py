# -*- coding: utf-8 -*-
"""
build_gui_demo.py  -  prepare the 'Multi-speaker demo' dataset for lip_compare_gui.py

Does three things:
  1. Copies the 8 chosen original-clip mp4s (speaker face, NOT ROI) into
     vsr_bench/gui/videos/multispeakerdemo/<utt_id>.mp4
  2. Filters T1 and B2 confA CSVs down to those 8 clips, saving them as
     vsr_bench/gui/results/T1_multispeakerdemo.csv  and  B2_multispeakerdemo.csv
     (the GUI's loader renames id -> utt_id automatically; normalize applies)
  3. Prints the one-line edit you need to make in lip_compare_gui.py CONFIG

Run:
  python build_gui_demo.py
"""
import csv
import os
import shutil

CACHE_MF = "/root/autodl-tmp/cache/multispeaker_test/manifest.csv"
RAW_CLIPS_ROOT = "/root/autodl-tmp/speaker_clips/test"
GUI_DIR = "/root/autodl-tmp/vsr_bench/gui"
VIDEOS_DIR = os.path.join(GUI_DIR, "videos", "multispeakerdemo")
RESULTS_DIR = os.path.join(GUI_DIR, "results")
CONF_DIR = "/root/autodl-tmp/day1/conf"

# ============== THE 8 DEMO CLIPS (manually curated from analysis) ==============
# Act 1: works well (WER < 20%)
# Act 2: partially correct (WER 20-30%)
# Act 3: failures showing when the system breaks
DEMO_IDS = [
    "nathaniel_drew__U-w7Koq_LkE__clip_0000",      # Act1 short 3.9s 11% - clean success
    "dave2d__CEgynoERrqc__clip_0007",              # Act1 medium 5.4s 12%
    "peter_mckinnon__fA3pFXwTnKw__clip_0004",      # Act1 medium 4.7s 17%
    "jack_edwards__z76kQ_hNGCI__clip_0152",        # Act2 medium 4.8s 20%
    "nathaniel_drew__yemarhfhVA0__clip_0225",      # Act2 medium 5.0s 20%
    "rachel_aust__ck-S90-WET4__clip_0002",         # Act2 medium 4.8s 22%
    "fireship___5p1_TNSWqQ__clip_0037",            # Act3 ROI failure (edited vlog)
    "mike_boyd__sxfAVFUtJ6A__clip_0016",           # Act3 model collapse ("dusty dusty dusty")
]

# =========================================================================


def main():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build utt_id -> (speaker, video, clip_id) map from manifest
    sp_map = {}
    with open(CACHE_MF, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp_map[r["utt_id"]] = (r["speaker"], r["video"])

    # ---- 1. Copy videos ----
    print("\n== 1. Copy original clips ==")
    missing_vid = []
    for uid in DEMO_IDS:
        if uid not in sp_map:
            print(f"  SKIP  {uid}  (not in manifest!)")
            missing_vid.append(uid)
            continue
        sp, vid = sp_map[uid]
        # clip_id is the last part after the last '__'
        clip_id = uid.rsplit("__", 1)[-1]
        src = os.path.join(RAW_CLIPS_ROOT, sp, vid, f"{clip_id}.mp4")
        dst = os.path.join(VIDEOS_DIR, f"{uid}.mp4")
        if not os.path.exists(src):
            print(f"  MISS  {src}")
            missing_vid.append(uid)
            continue
        shutil.copy2(src, dst)
        size_kb = os.path.getsize(dst) // 1024
        print(f"  ok    {uid}.mp4  ({size_kb} KB)")

    # ---- 2. Filter confA CSVs down to the demo ids ----
    print("\n== 2. Filter result CSVs ==")
    demo_set = set(DEMO_IDS)
    for model_id in ["T1", "B2"]:
        src_csv = os.path.join(CONF_DIR, f"{model_id}_multispeaker_test_confA.csv")
        dst_csv = os.path.join(RESULTS_DIR, f"{model_id}_multispeakerdemo.csv")
        if not os.path.exists(src_csv):
            print(f"  MISS  {src_csv}")
            continue
        with open(src_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Only keep columns the GUI cares about + confA_logp as 'seconds'-adjacent info
            # GUI uses: utt_id, ref, hyp, seconds (optional)
            # We keep: id, ref, hyp, seconds (if present)
            fields = ["id", "ref", "hyp", "seconds"]
            if "seconds" not in reader.fieldnames:
                fields = ["id", "ref", "hyp"]
            kept = [r for r in reader if r["id"] in demo_set]
        with open(dst_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
        print(f"  ok    {dst_csv}  ({len(kept)} rows)")

    # ---- 3. Report ----
    print("\n== 3. Final step: edit lip_compare_gui.py CONFIG ==\n")
    print('  In the CONFIG dict, change the "datasets" line to:')
    print('      "datasets": {')
    print('          "Multi-speaker demo": "multispeakerdemo",   # <-- NEW')
    print('          "BBC (Tom Scott)": "bbc",')
    print('          "Self-recorded": "self",')
    print('          "LRS3 test": "lrs3",')
    print('      },')
    print()
    print('  Also edit "default_models" to just show T1 and B2:')
    print('      "default_models": ["T1", "B2"],')
    print()
    print("Then launch the GUI as usual.  The new 'Multi-speaker demo' option")
    print("appears in the dataset dropdown on the 'Watch a clip' page.")
    if missing_vid:
        print(f"\nWARN: {len(missing_vid)} videos missing — check the clip paths:")
        for u in missing_vid:
            print(f"  - {u}")


if __name__ == "__main__":
    main()
