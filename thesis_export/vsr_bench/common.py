# -*- coding: utf-8 -*-
"""所有评测脚本共用：路径、文本规范化、WER、结果保存。保证所有模型用同一套规则。"""
import json
import os
import re

# ======== 路径（如果你的位置不同，只改这里）========
ROOT = os.environ.get("VSR_ROOT", "/root/autodl-tmp")   # 一般不用改
CODE_RAVEN = f"{ROOT}/code/raven"
CODE_AUTOAVSR = f"{ROOT}/code/auto_avsr"
CACHE = f"{ROOT}/cache"            # ROI 缓存：cache/{数据集}/xxx.npy + manifest.csv
RESULTS = f"{ROOT}/results_v2"     # 新的统一格式结果
LOGS = f"{ROOT}/logs"
HERE = os.path.dirname(os.path.abspath(__file__))

# ======== 文本规范化（和你之前的脚本完全一样：小写、去标点、数字转单词）========
_inflect = None


def normalize(text):
    global _inflect
    if _inflect is None:
        import inflect
        _inflect = inflect.engine()
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)

    def n2w(m):
        try:
            return _inflect.number_to_words(m.group(), andword="").replace("-", " ").replace(",", " ")
        except Exception:
            return m.group()

    text = re.sub(r"\d+", n2w, text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b):
    """词级编辑距离"""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def score(ref, hyp):
    """返回 (规范化 ref, 规范化 hyp, 错误数, 参考词数)"""
    r, h = normalize(ref), normalize(hyp)
    rw, hw = r.split(), h.split()
    return r, h, edit_distance(rw, hw), len(rw)


# ======== ROI 缓存 ========
def read_manifest(dataset):
    """cache/{dataset}/manifest.csv -> [(utt_id, text, npy路径)]"""
    import csv
    path = os.path.join(CACHE, dataset, "manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 prep_lrs3_cache.py 或 cache_bbc_rois.py")
    with open(path, encoding="utf-8") as f:
        return [(r["utt_id"], r["text"], os.path.join(CACHE, dataset, r["npy"])) for r in csv.DictReader(f)]


# ======== 结果保存（GUI 和汇总都读这个格式）========
def save_results(out_csv, rows, meta):
    """rows: dict 列表，至少有 utt_id, ref, hyp, seconds"""
    import pandas as pd
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    e, n = int(df["errors"].sum()), int(df["n_words"].sum())
    meta = dict(meta)
    meta.update({"n_clips": len(df), "errors": e, "n_words": n,
                 "wer": round(100.0 * e / max(n, 1), 2),
                 "mean_seconds": round(float(df["seconds"].mean()), 3) if len(df) else None,
                 "n_failed": int((df["hyp"].astype(str).str.len() == 0).sum())})
    with open(out_csv.replace(".csv", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("=" * 50)
    print(f"{meta.get('model_id', '')}  {meta.get('dataset', '')}")
    print(f"clips: {len(df)}   failed: {meta['n_failed']}   WER: {meta['wer']:.2f}%")
    print(f"mean seconds/clip: {meta['mean_seconds']}")
    print(f"saved: {out_csv}")
    print("=" * 50)
    return meta
