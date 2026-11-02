# -*- coding: utf-8 -*-
"""
fix_manifest_dedup.py  -  collapse repeated n-grams in cache manifest text

YouTube auto-caption pulled into the test manifest contains
contiguous repeated chunks like:
  "was a very interesting hardware story was a very interesting hardware story
   was a very interesting hardware story for them because ..."
This inflates REF word count 2-3x and makes WER meaningless.

Fix: for each text, scan left-to-right; at each position find the LARGEST n
such that words[i:i+n] == words[i+n:i+2n] (and keep extending if words[i:i+n]
repeats k times), then keep one copy and skip the rest.

Run:
  python fix_manifest_dedup.py \
    --in  /root/autodl-tmp/cache/multispeaker_test/manifest.csv \
    --out /root/autodl-tmp/cache/multispeaker_test/manifest.csv \
    --backup
"""
import argparse
import csv
import os
import re
import shutil


def dedup_repeated_ngrams(text, max_n=15):
    # normalize whitespace, keep original punctuation
    words = text.split()
    out = []
    i = 0
    n_words = len(words)
    while i < n_words:
        collapsed = False
        max_try = min(max_n, (n_words - i) // 2)
        for n in range(max_try, 0, -1):
            if words[i:i + n] == words[i + n:i + 2 * n]:
                # how many times does it repeat contiguously?
                k = 2
                while words[i:i + n] == words[i + k * n:i + (k + 1) * n]:
                    k += 1
                out.extend(words[i:i + n])
                i += k * n
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backup", action="store_true")
    ap.add_argument("--max-n", type=int, default=15)
    ap.add_argument("--preview", type=int, default=5,
                    help="show first N before/after examples")
    a = ap.parse_args()

    if a.backup and a.inp == a.out:
        bak = a.inp + ".bak"
        shutil.copyfile(a.inp, bak)
        print(f"backup -> {bak}")

    with open(a.inp, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    assert "text" in fields, f"text column missing. have: {fields}"

    total_before = total_after = 0
    changed = 0
    examples = []
    for r in rows:
        before = r["text"]
        after = dedup_repeated_ngrams(before, a.max_n)
        wb = len(before.split())
        wa = len(after.split())
        total_before += wb
        total_after += wa
        if wa < wb:
            changed += 1
            if len(examples) < a.preview:
                examples.append((r["utt_id"], before, after, wb, wa))
        r["text"] = after

    # write
    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, a.out)

    print(f"\nrows: {len(rows)}   changed: {changed}")
    print(f"total words:  {total_before} -> {total_after}  "
          f"(reduced {total_before - total_after}, "
          f"ratio {100 * total_after / max(1, total_before):.1f}%)")
    print(f"\n--- first {len(examples)} dedup examples ---")
    for uid, b, aft, wb, wa in examples:
        print(f"\n[{uid}]  {wb} -> {wa} words")
        print(f"  BEFORE: {b}")
        print(f"  AFTER : {aft}")


if __name__ == "__main__":
    main()
