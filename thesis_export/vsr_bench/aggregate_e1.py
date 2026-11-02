# -*- coding: utf-8 -*-
"""
aggregate_e1.py  -  E1 per-speaker + overall WER aggregation

Reads an nbest CSV written by eval_autoavsr_nbest.py / eval_braven_nbest.py
on dataset=multispeaker_test, joins back to speaker labels via the cache
manifest, then prints:
  - per-speaker WER (and clip count)
  - overall WER (micro, weighted by words)
  - macro mean / std across speakers (for E6 variance view)

Run:
  python aggregate_e1.py --csv /root/autodl-tmp/day1/conf/T1_multispeaker_test_confA.csv \
                         --model T1
  python aggregate_e1.py --csv /root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv \
                         --model B2
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, "/root/autodl-tmp/vsr_bench")
from common import CACHE, score  # score(hyp, ref) -> (wer, n_err, n_ref_words)

MANIFEST = os.path.join(CACHE, "multispeaker_test", "manifest.csv")


def load_speakers():
    """utt_id -> speaker from the cache manifest."""
    m = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[r["utt_id"]] = r["speaker"]
    return m


def wer(pairs):
    """pairs: list of (errors, n_words) -> (wer_pct, n_err, n_words, n_clips)."""
    tot_err = sum(int(e) for e, _ in pairs)
    tot_ref = sum(int(n) for _, n in pairs)
    return (100.0 * tot_err / max(1, tot_ref), tot_err, tot_ref, len(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="nbest confA csv from eval_*_nbest.py")
    ap.add_argument("--model", default="?", help="label (T1 / B2)")
    ap.add_argument("--hyp-col", default="hyp", help="hypothesis column name")
    ap.add_argument("--ref-col", default="ref", help="reference column name")
    ap.add_argument("--out", default="", help="optional per-speaker CSV output")
    a = ap.parse_args()

    sp_map = load_speakers()

    by_sp = {}
    with open(a.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # tolerate different column names
        cols = reader.fieldnames
        idcol = next((c for c in cols if c.lower() in ("utt_id", "id")), None)
        ecol = next((c for c in cols if c.lower() in ("errors", "n_err")), None)
        ncol = next((c for c in cols if c.lower() in ("n_words", "words", "n_ref")), None)
        if not (idcol and ecol and ncol):
            raise SystemExit(f"need id/errors/n_words columns. have: {cols}")
        for r in reader:
            sp = sp_map.get(r[idcol], "UNKNOWN")
            by_sp.setdefault(sp, []).append((r[ecol], r[ncol]))

    print(f"\n=== E1 [{a.model}] multispeaker_test ===")
    print(f"{'speaker':<22}{'clips':>8}{'words':>10}{'WER%':>10}")
    rows_out = []
    per_sp_wer = []
    all_pairs = []
    for sp in sorted(by_sp):
        pairs = by_sp[sp]
        w, ne, nw, nc = wer(pairs)
        print(f"{sp:<22}{nc:>8}{nw:>10}{w:>10.2f}")
        rows_out.append({"speaker": sp, "clips": nc, "words": nw, "wer_pct": f"{w:.2f}"})
        per_sp_wer.append(w)
        all_pairs.extend(pairs)

    ow, oe, onw, onc = wer(all_pairs)
    macro = statistics.mean(per_sp_wer)
    std = statistics.stdev(per_sp_wer) if len(per_sp_wer) > 1 else 0.0
    print("-" * 50)
    print(f"{'OVERALL (micro)':<22}{onc:>8}{onw:>10}{ow:>10.2f}")
    print(f"{'MACRO mean':<22}{'':>8}{'':>10}{macro:>10.2f}")
    print(f"{'MACRO std':<22}{'':>8}{'':>10}{std:>10.2f}")

    if a.out:
        rows_out.append({"speaker": "OVERALL_micro", "clips": onc, "words": onw, "wer_pct": f"{ow:.2f}"})
        rows_out.append({"speaker": "MACRO_mean", "clips": "", "words": "", "wer_pct": f"{macro:.2f}"})
        rows_out.append({"speaker": "MACRO_std", "clips": "", "words": "", "wer_pct": f"{std:.2f}"})
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["speaker", "clips", "words", "wer_pct"])
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
