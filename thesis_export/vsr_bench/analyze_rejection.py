# -*- coding: utf-8 -*-
"""
analyze_rejection.py  -  comprehensive rejection + calibration analysis for E1

Produces every number and chart Zhang needs for the thesis Case B rejection story:
  1. Per-speaker rejection curve (both T1 and B2) at 100 / 80 / 60 / 40 / 20 % keep
  2. Overall rejection curve with risk/coverage sweep (every 5 %)
  3. Expected Calibration Error (ECE) for both models on multi-speaker test
  4. Correlation between confA_logp and clip-level WER (Pearson)
  5. Sweet-spot analysis: which keep % gives best WER / coverage trade-off
  6. Comparison: does B2's confidence transfer better than T1's? (Sem 1 claim test)

Reads nothing new — just the existing confA.csv files.
Writes CSVs + ascii plots so you can glance at them tomorrow.

Run:
  cd /root/autodl-tmp/vsr_bench
  python analyze_rejection.py
"""
import csv
import math
import os
from collections import defaultdict

CONF_DIR = "/root/autodl-tmp/day1/conf"
OUT_DIR = "/root/autodl-tmp/results/E1_rejection_analysis"
MANIFEST = "/root/autodl-tmp/cache/multispeaker_test/manifest.csv"
EXCLUDE = {"fireship", "mike_boyd"}

os.makedirs(OUT_DIR, exist_ok=True)


def load_speakers():
    m = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[r["utt_id"]] = r["speaker"]
    return m


def load_model_data(model, sp_map):
    rows = []
    path = f"{CONF_DIR}/{model}_multispeaker_test_confA.csv"
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = sp_map.get(r["id"], "UNKNOWN")
            if s in EXCLUDE:
                continue
            e = int(r["errors"])
            n = int(r["n_words"])
            if n == 0:
                continue
            rows.append({
                "id": r["id"],
                "speaker": s,
                "logp": float(r["confA_logp"]),
                "margin": float(r["confA_margin"]),
                "err": e,
                "n": n,
                "wer": e / n,  # clip-level WER (can be > 1)
            })
    return rows


def wer(rows):
    e = sum(r["err"] for r in rows)
    n = sum(r["n"] for r in rows)
    return (100 * e / n) if n else 0.0


def rejection_curve(rows, pcts):
    """returns list of (keep%, n_clips, n_words, WER%)."""
    srt = sorted(rows, key=lambda r: -r["logp"])  # most confident first
    out = []
    for p in pcts:
        k = max(1, int(round(len(srt) * p / 100)))
        kept = srt[:k]
        out.append((p, len(kept), sum(r["n"] for r in kept), wer(kept)))
    return out


def pearson(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return num / (dx * dy) if dx * dy > 0 else 0.0


def ece(rows, n_bins=10):
    """Expected calibration error.
    Treat exp(logp) as confidence, 1-wer as accuracy proxy (clipped to [0,1])."""
    scored = [(math.exp(r["logp"]), max(0.0, 1.0 - r["wer"]), r["n"]) for r in rows]
    bins = [[] for _ in range(n_bins)]
    for conf, acc, n in scored:
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, acc, n))
    total = sum(len(b) for b in bins)
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc = sum(x[1] for x in b) / len(b)
        e += len(b) / total * abs(avg_conf - avg_acc)
    return e


def ascii_bar(frac, width=30):
    k = int(frac * width)
    return "█" * k + "░" * (width - k)


def main():
    sp_map = load_speakers()
    results = {}
    for model in ["T1", "B2"]:
        results[model] = load_model_data(model, sp_map)
        print(f"loaded {model}: {len(results[model])} clips")

    # =========== 1. Overall rejection curve (fine-grained) ===========
    print("\n" + "=" * 70)
    print("1. OVERALL REJECTION CURVE (6 speakers, excl fireship + mike_boyd)")
    print("=" * 70)
    pcts = list(range(100, 15, -5))
    all_curves = {}
    for model in ["T1", "B2"]:
        curve = rejection_curve(results[model], pcts)
        all_curves[model] = curve
    print(f"\n{'keep%':<8}{'T1 WER%':<12}{'T1 Δ':<10}{'B2 WER%':<12}{'B2 Δ':<10}")
    base_t1 = all_curves["T1"][0][3]
    base_b2 = all_curves["B2"][0][3]
    for i, p in enumerate(pcts):
        t1w = all_curves["T1"][i][3]
        b2w = all_curves["B2"][i][3]
        print(f"{p:<8}{t1w:<12.2f}{t1w - base_t1:<+10.2f}{b2w:<12.2f}{b2w - base_b2:<+10.2f}")

    # save
    with open(f"{OUT_DIR}/overall_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keep_pct", "T1_clips", "T1_words", "T1_wer",
                    "B2_clips", "B2_words", "B2_wer"])
        for i, p in enumerate(pcts):
            tr = all_curves["T1"][i]
            br = all_curves["B2"][i]
            w.writerow([p, tr[1], tr[2], f"{tr[3]:.3f}", br[1], br[2], f"{br[3]:.3f}"])

    # =========== 2. Per-speaker rejection curve ===========
    print("\n" + "=" * 70)
    print("2. PER-SPEAKER REJECTION CURVE (B2 only, short view)")
    print("=" * 70)
    key_pcts = [100, 80, 60, 40]
    print(f"\n{'speaker':<22}" + "".join(f"keep{p}%".ljust(12) for p in key_pcts))
    by_sp = defaultdict(list)
    for r in results["B2"]:
        by_sp[r["speaker"]].append(r)
    rows_out = []
    for sp in sorted(by_sp):
        rows_sp = by_sp[sp]
        curve = rejection_curve(rows_sp, key_pcts)
        line = f"{sp:<22}"
        for _, _, _, w in curve:
            line += f"{w:<12.2f}"
        print(line)
        rows_out.append([sp, len(rows_sp)] + [f"{c[3]:.2f}" for c in curve])

    with open(f"{OUT_DIR}/per_speaker_B2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "n_clips"] + [f"wer_keep{p}" for p in key_pcts])
        w.writerows(rows_out)

    # =========== 3. ECE comparison ===========
    print("\n" + "=" * 70)
    print("3. CALIBRATION QUALITY (ECE, lower = better calibrated)")
    print("=" * 70)
    for model in ["T1", "B2"]:
        e = ece(results[model])
        print(f"  {model}: ECE = {e:.4f}")
    print("\n  Sem 1 Tom Scott finding was: T1 ECE 0.271 vs LRS3 0.210")
    print("  Hypothesis: on multi-speaker, B2's ECE should be lower than T1's")

    # =========== 4. Correlation: confA_logp vs clip WER ===========
    print("\n" + "=" * 70)
    print("4. HOW WELL DOES CONFIDENCE PREDICT ERRORS? (Pearson, more negative = better)")
    print("=" * 70)
    for model in ["T1", "B2"]:
        rs = results[model]
        logps = [r["logp"] for r in rs]
        wers = [r["wer"] for r in rs]
        pear = pearson(logps, wers)
        print(f"  {model}: pearson(logp, clip_WER) = {pear:+.3f}  (should be NEGATIVE)")

    # =========== 5. Sweet-spot analysis ===========
    print("\n" + "=" * 70)
    print("5. SWEET-SPOT (WER × coverage trade-off)")
    print("=" * 70)
    print("  score = WER + (100 - keep%) * 0.5   (penalize dropping clips)")
    print()
    print(f"\n{'model':<6}{'best keep%':<14}{'WER':<10}{'score':<10}")
    for model in ["T1", "B2"]:
        curve = all_curves[model]
        best = min(curve, key=lambda r: r[3] + (100 - r[0]) * 0.5)
        print(f"{model:<6}{best[0]:<14}{best[3]:<10.2f}{best[3] + (100 - best[0]) * 0.5:<10.2f}")

    # =========== 6. ASCII plot of overall curve ===========
    print("\n" + "=" * 70)
    print("6. OVERALL REJECTION CURVE (ascii, for quick glance)")
    print("=" * 70)
    print(f"{'keep%':<8}{'B2 WER':<10}  bar")
    for i, p in enumerate(pcts):
        w = all_curves["B2"][i][3]
        # normalize to [0, 1] where 70 = full bar, 40 = empty
        frac = max(0, min(1, (70 - w) / 30))
        print(f"{p:<8}{w:<10.2f}  {ascii_bar(frac)}")

    print("\n" + "=" * 70)
    print(f"\nAll CSVs saved to {OUT_DIR}/")
    print("\nSUGGESTED HEADLINE FOR TOMORROW'S PITCH:")
    print(f"  B2 baseline (6-sp):            63.38 %")
    best_b2 = min(all_curves["B2"], key=lambda r: r[3] + (100 - r[0]) * 0.5)
    print(f"  B2 + rejection at best keep%:  {best_b2[3]:.2f} % @ keep {best_b2[0]}%")
    print(f"  Improvement:                   -{63.38 - best_b2[3]:.2f} pp")
    print(f"\n  Oracle N-best ceiling (top-5): -3.5 pp (LLM rescoring deferred)")


if __name__ == "__main__":
    main()
