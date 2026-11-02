#!/usr/bin/env python
"""统一文本归一化 + 重算 WER（应用到所有模型结果）"""
import sys, re
import pandas as pd
import torchaudio
import inflect

p = inflect.engine()

def normalize(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    def num_to_words(m):
        try:
            return p.number_to_words(m.group(), andword='').replace('-', ' ').replace(',', ' ')
        except:
            return m.group()
    text = re.sub(r'\d+', num_to_words, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compute_wer(csv_path):
    df = pd.read_csv(csv_path)
    raw_dist, raw_words = 0, 0
    norm_dist, norm_words = 0, 0
    for _, r in df.iterrows():
        ref, hyp = str(r['ref']), str(r['hyp'])
        # raw
        rw, hw = ref.lower().split(), hyp.lower().split()
        raw_dist += torchaudio.functional.edit_distance(rw, hw)
        raw_words += len(rw)
        # normalized
        nref, nhyp = normalize(ref).split(), normalize(hyp).split()
        norm_dist += torchaudio.functional.edit_distance(nref, nhyp)
        norm_words += len(nref)
    return 100*raw_dist/raw_words, 100*norm_dist/norm_words

if __name__ == "__main__":
    for name, path in [
        ("Auto-AVSR LRS3", "/root/autodl-tmp/results/autoavsr_lrs3_full.csv"),
        ("Auto-AVSR BBC",  "/root/autodl-tmp/results/autoavsr_bbc_full.csv"),
    ]:
        try:
            raw, norm = compute_wer(path)
            print(f"{name}: raw={raw:.2f}%  normalized={norm:.2f}%")
        except Exception as e:
            print(f"{name}: ERROR {e}")
