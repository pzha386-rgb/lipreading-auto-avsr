"""BBC ROI manifest -> Auto-AVSR eval.py CSV (LRS3 format, 4 cols)"""
import csv
import os

SPM_MODEL = "/data/default/auto_avsr/spm/unigram/unigram5000.model"
UNITS_FILE = "/data/default/auto_avsr/spm/unigram/unigram5000_units.txt"
MANIFEST_IN = "/data/default/bbc_test/manifest_rois.csv"
LABELS_DIR = "/data/default/bbc_test/labels"
CSV_OUT = "/data/default/bbc_test/labels/bbc_test.csv"
DATASET_NAME = "bbc"

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=SPM_MODEL)
with open(UNITS_FILE, encoding="utf8") as f:
    units = f.read().splitlines()
hashmap = {u.split()[0]: u.split()[-1] for u in units}
unk_id = hashmap["<unk>"]
print(f"Vocab size: {len(hashmap)}, <unk> id: {unk_id}")

def tokenize(text):
    text = text.upper()
    cleaned = ''.join(c if (c.isalnum() or c.isspace() or c == "'") else ' ' for c in text)
    cleaned = ' '.join(cleaned.split())
    pieces = sp.EncodeAsPieces(cleaned)
    ids = [hashmap.get(p, unk_id) for p in pieces]
    return ids, pieces, cleaned

os.makedirs(LABELS_DIR, exist_ok=True)
with open(MANIFEST_IN) as f:
    rows = list(csv.DictReader(f))
print(f"Read {len(rows)} rows\n")

out_lines = []
total_pieces = 0
total_unk = 0
samples = []

for row in rows:
    roi_frames = int(row['roi_frames'])
    text = row['text']
    rel_path = f"rois/{row['path']}"
    ids, pieces, cleaned = tokenize(text)
    total_pieces += len(pieces)
    total_unk += sum(1 for p in pieces if hashmap.get(p, unk_id) == unk_id)
    line = f"{DATASET_NAME},{rel_path},{roi_frames},{' '.join(ids)}"
    out_lines.append(line)
    if len(samples) < 3:
        samples.append((row['clip_id'], text[:60], cleaned[:60], pieces[:10], ids[:10]))

with open(CSV_OUT, 'w') as f:
    for line in out_lines:
        f.write(line + '\n')

print("=== Sample (first 3) ===")
for cid, orig, upper, pieces, ids in samples:
    print(f"\n[{cid}]")
    print(f"  Orig:   {orig}")
    print(f"  Upper:  {upper}")
    print(f"  Pieces: {pieces}")
    print(f"  IDs:    {ids}")

print(f"\n=== Stats ===")
print(f"Clips: {len(out_lines)}")
print(f"Pieces: {total_pieces}, UNK: {total_unk} ({total_unk/max(total_pieces,1)*100:.2f}%)")
print(f"  → 应该 < 1%。如果 > 5% 报告里要解释")
print(f"\nCSV: {CSV_OUT}")
print(f"\nFirst 3 lines:")
for line in out_lines[:3]:
    print(f"  {line[:200]}")
