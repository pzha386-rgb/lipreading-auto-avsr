# -*- coding: utf-8 -*-
"""
build_gui_demo_v2.py  -  same as v1 but keeps confA_logp / confA_margin columns
needed by the Robot Scenario GUI.

Also writes a condensed robot_scenario.csv with just the 3 chosen users for the
scenario demo (one clip each).

Run:
  python build_gui_demo_v2.py
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

DEMO_IDS = [
    "nathaniel_drew__U-w7Koq_LkE__clip_0000",      # ACCEPT  (short, clean)
    "dave2d__CEgynoERrqc__clip_0007",
    "peter_mckinnon__fA3pFXwTnKw__clip_0004",
    "jack_edwards__z76kQ_hNGCI__clip_0152",        # ASK AGAIN
    "nathaniel_drew__yemarhfhVA0__clip_0225",
    "rachel_aust__ck-S90-WET4__clip_0002",
    "fireship___5p1_TNSWqQ__clip_0037",            # REJECT
    "mike_boyd__sxfAVFUtJ6A__clip_0016",
]

# The 3 chosen for the "robot scenario" story
SCENARIO_IDS = [
    "nathaniel_drew__U-w7Koq_LkE__clip_0000",   # Act1: ACCEPT
    "jack_edwards__z76kQ_hNGCI__clip_0152",     # Act2: ASK AGAIN
    "fireship___5p1_TNSWqQ__clip_0037",         # Act3: REJECT
]


def main():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sp_map = {}
    with open(CACHE_MF, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp_map[r["utt_id"]] = (r["speaker"], r["video"])

    # ---- 1. videos (idempotent) ----
    print("\n== 1. Videos (skipping if exists) ==")
    for uid in DEMO_IDS:
        if uid not in sp_map:
            print(f"  SKIP  {uid}")
            continue
        sp, vid = sp_map[uid]
        clip_id = uid.rsplit("__", 1)[-1]
        src = os.path.join(RAW_CLIPS_ROOT, sp, vid, f"{clip_id}.mp4")
        dst = os.path.join(VIDEOS_DIR, f"{uid}.mp4")
        if os.path.exists(dst):
            print(f"  skip  {uid}  (already copied)")
            continue
        if not os.path.exists(src):
            print(f"  MISS  {src}")
            continue
        shutil.copy2(src, dst)
        print(f"  ok    {uid}.mp4")

    # ---- 2. result CSVs — this time KEEP confA_logp / confA_margin ----
    print("\n== 2. Result CSVs (with confidence columns) ==")
    demo_set = set(DEMO_IDS)
    for model_id in ["T1", "B2"]:
        src_csv = os.path.join(CONF_DIR, f"{model_id}_multispeaker_test_confA.csv")
        dst_csv = os.path.join(RESULTS_DIR, f"{model_id}_multispeakerdemo.csv")
        if not os.path.exists(src_csv):
            print(f"  MISS  {src_csv}")
            continue
        with open(src_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            keep_cols = ["id", "ref", "hyp", "seconds", "confA_logp", "confA_margin", "errors", "n_words"]
            fields = [c for c in keep_cols if c in reader.fieldnames]
            kept = [r for r in reader if r["id"] in demo_set]
        with open(dst_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
        print(f"  ok    {dst_csv}  ({len(kept)} rows, {len(fields)} cols)")

    # ---- 3. Scenario CSV for the robot GUI (B2 model, 3 chosen clips) ----
    print("\n== 3. Robot Scenario CSV ==")
    scenario_csv = os.path.join(RESULTS_DIR, "robot_scenario.csv")
    scn_set = set(SCENARIO_IDS)
    with open(f"{CONF_DIR}/B2_multispeaker_test_confA.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["id"] in scn_set]
    # Order rows to match SCENARIO_IDS order (ACCEPT -> ASK -> REJECT)
    rows_sorted = sorted(rows, key=lambda r: SCENARIO_IDS.index(r["id"]))
    # Add scenario-friendly columns
    for i, r in enumerate(rows_sorted):
        sp, vid = sp_map[r["id"]]
        r["speaker_display"] = sp.replace("_", " ").title()
        r["step"] = i + 1
    fields = ["step", "id", "speaker_display", "ref", "hyp", "confA_logp", "confA_margin", "seconds"]
    with open(scenario_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_sorted)
    print(f"  ok    {scenario_csv}  ({len(rows_sorted)} rows)")
    print()
    print("Scenario clips in order:")
    for r in rows_sorted:
        print(f"  {r['step']}. {r['speaker_display']:<20}  logp={float(r['confA_logp']):+.2f}  '{r['ref'][:50]}...'")


if __name__ == "__main__":
    main()
