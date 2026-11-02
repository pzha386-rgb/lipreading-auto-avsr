# -*- coding: utf-8 -*-
"""
唇语模型对比 GUI（用于组会 / 毕业答辩演示）

三页：
  1. Watch a clip   选一个视频 -> 播放 -> 各模型识别结果并排，错词标红
  2. Key findings   只放 3 个核心发现（每个一张图 + 一句话）
  3. Live test      （可选）上传视频，调用你现有的 FastAPI 后端

需要的文件（相对本脚本所在目录，可在 CONFIG 修改）：
  models.csv                     模型清单（和评测用的是同一个）
  results/{模型编号}_{数据集}.csv  每个模型的识别结果
      列名：utt_id, ref, hyp, seconds   （wer 列可有可无，会自动重算）
  videos/{数据集}/{utt_id}.mp4     原始人脸视频（用来播放）

运行：
  pip install gradio pandas matplotlib requests
  python lip_compare_gui.py            # 正式运行
  python lip_compare_gui.py --demo     # 生成假数据试运行（检查界面用）
"""
import argparse
import html
import os
import random

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ============================ CONFIG ============================
def _find_data():
    """找导出的 gui_data 文件夹（解压 gui_data.zip 得到）"""
    for d in [os.environ.get("GUI_DATA", ""), os.path.join(HERE, "gui_data"),
              os.path.join(os.path.dirname(HERE), "gui_data"),
              os.path.join(os.path.dirname(os.path.dirname(HERE)), "gui_data")]:
        if d and os.path.isdir(d):
            return d
    return os.path.join(HERE, "gui_data")


DATA = _find_data()
CONFIG = {
    "models_csv": os.path.join(DATA, "models.csv"),
    "results_dir": os.path.join(DATA, "results"),
    "videos_dir": os.path.join(DATA, "videos"),
    "figures_dir": os.path.join(DATA, "figures"),
    # 数据集：显示名 -> 文件里用的名字
    "datasets": {"BBC — new speaker, out-of-domain": "bbc", "LRS3 — TED talks, in-domain (mouth crop)": "lrs3_300"},
    # 默认只显示这 6 个代表模型（老师不想看太多）
    "default_models": ["S0", "T1", "T3", "B4", "U4", "U4g"],
    "display_names": {
        "S6": "Auto-AVSR (23 h labels)", "S0": "Auto-AVSR (438 h labels)", "S1": "Auto-AVSR (1,759 h)",
        "T1": "Auto-AVSR (3,291 h labels)", "S2": "Auto-AVSR (VSR-ML)", "S2lm": "Auto-AVSR (VSR-ML) + LM",
        "S4": "Ma 2022 (LRS3)", "S4lm": "Ma 2022 (LRS3) + LM", "S5": "Ma 2022 (LRS2)", "S5lm": "Ma 2022 (LRS2) + LM",
        "T3": "BRAVEn (30 h labels + ST)", "B1": "BRAVEn (30 h labels)", "B2": "BRAVEn (30 h + ST) + LM",
        "B3": "BRAVEn (433 h labels)", "B4": "BRAVEn (433 h labels + ST)", "B5": "BRAVEn (433 h + ST) + LM",
        "T4": "RAVEn (30 h + ST)", "R1": "RAVEn (30 h + ST) + LM", "R2": "RAVEn (433 h + ST)", "R3": "RAVEn (433 h + ST) + LM",
        "U1": "USR 2.0 Large (30 h)", "U2": "USR 2.0 Base+", "U3": "USR 2.0 Large", "U4": "USR 2.0 Huge",
        "U1g": "USR 2.0 Large (30 h, fast)", "U2g": "USR 2.0 Base+ (fast)", "U3g": "USR 2.0 Large (fast)",
        "U4g": "USR 2.0 Huge (fast)", "U4b10": "USR 2.0 Huge (beam 10)",
    },
    # Key findings 页：论文图（make_figures.py 生成）+ 一句话说明
    "findings": [
        ("fig1_label_efficiency.png",
         "1. Self-supervised models reach the same accuracy with 30 hours of labels "
         "that supervised models need 3,291 hours for."),
        ("fig3_in_vs_out_domain.png",
         "2. The best model on the benchmark (LRS3) is not the best on a new speaker (BBC) — "
         "testing on new data matters."),
        ("fig5_speed_vs_accuracy.png",
         "3. Decoding strategy matters more than model size for speed: the largest model with fast "
         "decoding is the quickest of all."),
        ("fig6_robustness.png",
         "4. For a robot camera, a stable frame rate matters most; darkness barely matters."),
    ],
    # 下面三项只在找不到论文图时使用
    "finding1": {"supervised": ["S6"], "self-supervised": ["T3", "T4", "U1"]},
    "finding2": [("S6", 23), ("S0", 438), ("S1", 1759), ("T1", 3291)],
    "finding_dataset": "lrs3_300",
    # 实时识别后端（你现有的 FastAPI，通过 Cloudflare tunnel）
    "backend_url": "http://127.0.0.1:7001",
    # Live test：每个服务一个模型（不同端口，SSH 隧道一起转发）
    "backends": {
        "Supervised · Auto-AVSR (3,291 h labels)": "http://127.0.0.1:7001",
        "Self-supervised · BRAVEn (30 h labels)": "http://127.0.0.1:7002",
        "Self-supervised · USR 2.0 Huge (fast)": "http://127.0.0.1:7003",
    },
}

# 颜色：监督 = 蓝，自监督 = 绿松石，LLM = 紫；错误 = 深红
TYPE_COLOR = {"supervised": "#2F5D8A", "ssl": "#1F8A70", "llm": "#7A4E9E"}
ERR_RED = "#B3263E"


def type_group(t):
    t = str(t)
    if "llm" in t:
        return "llm"
    if t.startswith("supervised"):
        return "supervised"
    return "ssl"


TYPE_LABEL = {"supervised": "Supervised", "ssl": "Self-supervised", "llm": "LLM-based"}

# ============================ 数据读取 ============================
try:                       # 和评测脚本用同一个规范化函数
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(HERE))
    from common import normalize as NORMALIZE
except Exception:
    NORMALIZE = None

MODELS = None      # DataFrame，按 id 索引
RESULTS = {}       # (model_id, dataset) -> DataFrame


def load_all():
    global MODELS, RESULTS
    MODELS = pd.read_csv(CONFIG["models_csv"], dtype=str).fillna("")
    MODELS["group"] = MODELS["type"].apply(type_group)
    MODELS = MODELS.set_index("id")
    RESULTS = {}
    rdir = CONFIG["results_dir"]
    if not os.path.isdir(rdir):
        return
    for fn in os.listdir(rdir):
        if not fn.endswith(".csv") or "_" not in fn:
            continue
        mid, ds = fn[:-4].split("_", 1)      # 例如 U4_bbc_blur2 -> U4, bbc_blur2
        if mid not in MODELS.index or "_limit" in fn:
            continue
        df = pd.read_csv(os.path.join(rdir, fn), dtype={"utt_id": str}).fillna("")
        df = df.rename(columns={"id": "utt_id", "clip_id": "utt_id", "reference": "ref",
                                "hypothesis": "hyp", "time": "seconds", "耗时": "seconds"})
        if not {"utt_id", "ref", "hyp"} <= set(df.columns):
            continue
        # 统一用规范化后的文本对比（数字转单词），和评测脚本的 WER 一致
        if {"ref_norm", "hyp_norm"} <= set(df.columns):
            df["ref"], df["hyp"] = df["ref_norm"].astype(str), df["hyp_norm"].astype(str)
        elif NORMALIZE:
            df["ref"] = df["ref"].astype(str).map(NORMALIZE)
            df["hyp"] = df["hyp"].astype(str).map(NORMALIZE)
        RESULTS[(mid, ds)] = df.set_index("utt_id")


def name_of(mid):
    if mid in CONFIG["display_names"]:
        return CONFIG["display_names"][mid]
    row = MODELS.loc[mid]
    return f"{row['family']} ({mid})"


def models_with(ds):
    return [m for m in MODELS.index if (m, ds) in RESULTS]


# ============================ 词级对齐 ============================
def norm_words(s):
    return [w for w in str(s).lower().replace(",", " ").replace(".", " ").split() if w]


def align(ref, hyp):
    """编辑距离对齐，返回 [(op, ref_word, hyp_word)]，op ∈ ok/sub/del/ins"""
    r, h = norm_words(ref), norm_words(hyp)
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("ok" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", r[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, h[j - 1]))
            j -= 1
    return ops[::-1]


def errors_and_len(ref, hyp):
    ops = align(ref, hyp)
    return sum(op != "ok" for op, _, _ in ops), max(len(norm_words(ref)), 1)


def corpus_wer(mid, ds):
    df = RESULTS.get((mid, ds))
    if df is None or len(df) == 0:
        return None
    e = n = 0
    for _, row in df.iterrows():
        a, b = errors_and_len(row["ref"], row["hyp"])
        e, n = e + a, n + b
    return 100.0 * e / n


def mean_seconds(mid, ds):
    df = RESULTS.get((mid, ds))
    if df is None or "seconds" not in df.columns:
        return None
    s = pd.to_numeric(df["seconds"], errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


# ============================ 第 1 页：看一个视频 ============================
def hyp_html(ops):
    out = []
    for op, rw, hw in ops:
        if op == "ok":
            out.append(f'<span class="w ok">{html.escape(hw)}</span>')
        elif op == "sub":
            out.append(f'<span class="w sub" title="should be: {html.escape(rw)}">{html.escape(hw)}</span>')
        elif op == "ins":
            out.append(f'<span class="w ins" title="extra word">{html.escape(hw)}</span>')
        else:
            out.append(f'<span class="w del" title="missed word">{html.escape(rw)}</span>')
    return " ".join(out) if out else '<span class="w del">(no output)</span>'


def clip_choices(ds_label):
    ds = CONFIG["datasets"][ds_label]
    ids = set()
    for m in models_with(ds):
        ids |= set(RESULTS[(m, ds)].index)
    return sorted(ids)


def show_clip(ds_label, utt, show_all):
    ds = CONFIG["datasets"][ds_label]
    if not utt:
        return None, '<p class="empty">Pick a clip to start.</p>'
    vpath = os.path.join(CONFIG["videos_dir"], ds, f"{utt}.mp4")
    video = vpath if os.path.exists(vpath) else None

    mids = models_with(ds)
    if not show_all:
        mids = [m for m in CONFIG["default_models"] if m in mids]
    ref = ""
    rows = []
    for m in mids:
        df = RESULTS[(m, ds)]
        if utt not in df.index:
            continue
        r = df.loc[utt]
        ref = r["ref"]
        ops = align(r["ref"], r["hyp"])
        e, n = errors_and_len(r["ref"], r["hyp"])
        g = MODELS.loc[m, "group"]
        rows.append((e / n, m, g, ops, e, n))
    rows.sort(key=lambda x: x[0])     # 错得少的排前面

    parts = [f'<div class="ref"><div class="ref-label">What was said</div>'
             f'<div class="ref-text">{html.escape(ref) or "-"}</div></div>']
    if not video:
        parts.append(f'<p class="note">Video file not found: videos/{ds}/{html.escape(utt)}.mp4</p>')
    for wer, m, g, ops, e, n in rows:
        parts.append(
            f'<div class="row" style="border-left-color:{TYPE_COLOR[g]}">'
            f'<div class="row-head"><span class="mname">{html.escape(name_of(m))}</span>'
            f'<span class="mtype" style="color:{TYPE_COLOR[g]}">{TYPE_LABEL[g]}</span>'
            f'<span class="mwer">{e} of {n} words wrong</span></div>'
            f'<div class="hyp">{hyp_html(ops)}</div></div>')
    if not rows:
        parts.append('<p class="empty">No results for this clip yet.</p>')
    parts.append('<p class="legend"><span class="w sub">wrong word</span> '
                 '<span class="w del">missed word</span> <span class="w ins">extra word</span> '
                 '· hover a red word to see the right one</p>')
    return video, "".join(parts)


# ============================ 第 2 页：三个发现 ============================
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)


def fig_finding1():
    ds = CONFIG["finding_dataset"]
    labels, vals, cols = [], [], []
    for grp, mids in CONFIG["finding1"].items():
        for m in mids:
            w = corpus_wer(m, ds)
            if w is None:
                continue
            labels.append(name_of(m))
            vals.append(w)
            cols.append(TYPE_COLOR["supervised" if grp == "supervised" else "ssl"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if vals:
        bars = ax.barh(labels, vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, max(105, max(vals) + 10))
    ax.set_xlabel("Word error rate (%)  — lower is better")
    _style(ax)
    fig.tight_layout()
    return fig


def fig_finding2():
    ds = CONFIG["finding_dataset"]
    pts = [(h, corpus_wer(m, ds), m) for m, h in CONFIG["finding2"]]
    pts = [p for p in pts if p[1] is not None]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if pts:
        xs, ys, ms = zip(*pts)
        ax.plot(xs, ys, "-o", color=TYPE_COLOR["supervised"], lw=2, ms=8)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=11)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:,} h" for x in xs])
    ax.set_xlabel("Hours of labelled training video")
    ax.set_ylabel("Word error rate (%)")
    _style(ax)
    fig.tight_layout()
    return fig


def fig_finding3():
    ds = CONFIG["finding_dataset"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models_with(ds):
        w, s = corpus_wer(m, ds), mean_seconds(m, ds)
        if w is None or s is None:
            continue
        g = MODELS.loc[m, "group"]
        ax.scatter(s, w, s=70, color=TYPE_COLOR[g], label=TYPE_LABEL[g], zorder=3)
        if m in CONFIG["default_models"]:          # 只标注代表模型，避免文字重叠
            ax.annotate(name_of(m), (s, w), textcoords="offset points", xytext=(6, 4), fontsize=9)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    if uniq:
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=10)
    ax.set_xlabel("Seconds per clip  — lower is faster")
    ax.set_ylabel("Word error rate (%)")
    ax.grid(alpha=0.25)
    _style(ax)
    fig.tight_layout()
    return fig


# ============================ 第 3 页：实时识别 ============================
def _health(url):
    import requests
    try:
        return requests.get(url.rstrip("/") + "/health", timeout=5).json()
    except Exception:
        return None


def check_servers():
    rows = []
    for label, url in CONFIG["backends"].items():
        h = _health(url)
        rows.append(f"- **{label}** ({url}): " + (f"✅ {h['model_name']} on {h['device']}" if h else "❌ not reachable"))
    return "\n".join(rows)


def _ask(url, video_path):
    import time
    import requests
    with open(video_path, "rb") as f:
        job = requests.post(f"{url}/submit", files={"file": f}, timeout=120).json()["job_id"]
    for _ in range(240):
        s = requests.get(f"{url}/status/{job}", timeout=30).json()
        if s.get("status") in ("done", "error"):
            return s
        time.sleep(0.5)
    return {"status": "error", "error": "timeout"}


def live_test(video_path, reference, chosen):
    from concurrent.futures import ThreadPoolExecutor
    if not video_path:
        return '<p class="empty">Upload or record a video first.</p>'
    targets = [(l, CONFIG["backends"][l]) for l in chosen if l in CONFIG["backends"]]
    if not targets:
        return '<p class="empty">Tick at least one model.</p>'
    with ThreadPoolExecutor(len(targets)) as ex:
        res = list(ex.map(lambda t: (t[0], _safe_ask(t[1], video_path)), targets))
    ref = (reference or "").strip()
    ref_n = NORMALIZE(ref) if (ref and NORMALIZE) else ref.lower()
    parts = []
    if ref:
        parts.append(f'<div class="ref"><div class="ref-label">What was said</div>'
                     f'<div class="ref-text">{html.escape(ref)}</div></div>')
    for label, r in res:
        g = "supervised" if label.startswith("Supervised") else "ssl"
        if r.get("status") != "done":
            body = f'<span class="w del">{html.escape(str(r.get("error", "failed")))}</span>'
            head = ""
        else:
            hyp = r["text"]
            if ref:
                hyp_n = NORMALIZE(hyp) if NORMALIZE else hyp.lower()
                ops = align(ref_n, hyp_n)
                e, n = errors_and_len(ref_n, hyp_n)
                body = hyp_html(ops)
                head = f"{e} of {n} words wrong · "
            else:
                body = html.escape(hyp or "(no words recognised)")
                head = ""
            t = r["timing"]
            head += f"{t['total']} s (read video {t['decode']} + face crop {t['face']} + model {t['infer']})"
        parts.append(f'<div class="row" style="border-left-color:{TYPE_COLOR[g]}">'
                     f'<div class="row-head"><span class="mname">{html.escape(label)}</span>'
                     f'<span class="mwer">{head}</span></div><div class="hyp">{body}</div></div>')
    if ref:
        parts.append('<p class="legend"><span class="w sub">wrong word</span> '
                     '<span class="w del">missed word</span> <span class="w ins">extra word</span></p>')
    return "".join(parts)


def _safe_ask(url, video_path):
    try:
        return _ask(url.rstrip("/"), video_path)
    except Exception as e:
        return {"status": "error", "error": f"cannot reach {url} ({type(e).__name__})"}


# ============================ 界面 ============================
CSS = """
.gradio-container {max-width: 1240px !important}
#title h1 {font-size: 26px; margin: 2px 0 0 0; font-weight: 600; letter-spacing: -0.01em}
#title p {margin: 2px 0 0 0; color: #4A5568; font-size: 14px}
.ref {background:#EEF2F6; border-radius:8px; padding:8px 14px; margin-bottom:8px}
.ref-label {font-size:12px; color:#4A5568; margin-bottom:2px}
.ref-text {font-size:18px; line-height:1.35; color:#1A2433}
.row {border-left:5px solid; padding:5px 12px; margin:5px 0; background:#FFFFFF;
      border-radius:0 8px 8px 0; box-shadow: 0 1px 0 #E3E8EE}
.row-head {display:flex; gap:12px; align-items:baseline; margin-bottom:1px}
.mname {font-weight:600; font-size:14px; color:#1A2433}
.mtype {font-size:12px}
.mwer {margin-left:auto; font-size:12px; color:#4A5568}
.hyp {font-size:16px; line-height:1.45}
.w {padding:0 2px; border-radius:3px}
.w.ok {color:#1A2433}
.w.sub {background:#F9DDE2; color:%(red)s; font-weight:600; cursor:help}
.w.ins {color:%(red)s; text-decoration: underline wavy; cursor:help}
.w.del {color:%(red)s; text-decoration: line-through; opacity:.75; cursor:help}
.legend {font-size:12px; color:#4A5568; margin-top:6px}
.note, .empty {color:#4A5568; font-size:14px}
.finding {font-size:17px; color:#1A2433; margin: 6px 0 2px 0}
.live-text textarea {font-size:22px !important}
""" % {"red": ERR_RED}

# 录制模式：字体放大，录屏后在投影上也看得清
CSS_BIG = """
.ref-text {font-size:20px !important}
.hyp {font-size:18px !important}
.mname {font-size:15px !important}
.mtype, .mwer {font-size:13px !important}
#title h1 {font-size:28px !important}
.finding {font-size:18px !important}
"""


GR_MAJOR = int(gr.__version__.split(".")[0])


def make_theme():
    return gr.themes.Base(primary_hue="blue", neutral_hue="slate",
                          font=[gr.themes.GoogleFont("IBM Plex Sans"), "Arial", "sans-serif"])


def build_ui():
    ds_labels = list(CONFIG["datasets"].keys())
    first_ds = next((d for d in ds_labels if clip_choices(d)), ds_labels[0])
    first_clips = clip_choices(first_ds)
    n_models = len({m for (m, _) in RESULTS})

    extra = {} if GR_MAJOR >= 6 else {"theme": make_theme(), "css": globals()["CSS"]}
    with gr.Blocks(title="Lip reading model comparison", **extra) as demo:
        gr.HTML(f'<div id="title"><h1>Reading lips: which model gets it right?</h1>'
                f'<p>{n_models} models compared on the same videos. '
                f'Blue = supervised, green = self-supervised, purple = LLM-based.</p></div>')

        with gr.Tab("Watch a clip"):
            with gr.Row():
                with gr.Column(scale=4):
                    ds = gr.Dropdown(ds_labels, value=first_ds, label="Video set")
                    clip = gr.Dropdown(first_clips, value=first_clips[0] if first_clips else None,
                                       label="Clip")
                    with gr.Row():
                        prev_b = gr.Button("Previous clip")
                        next_b = gr.Button("Next clip", variant="primary")
                    show_all = gr.Checkbox(False, label="Show all models (default: 6 key models)")
                    video = gr.Video(label="Video", height=260, interactive=False, autoplay=True)
                with gr.Column(scale=6):
                    out = gr.HTML()

            def on_ds(d):
                c = clip_choices(d)
                return gr.update(choices=c, value=c[0] if c else None)

            def step(d, u, k):
                c = clip_choices(d)
                if not c:
                    return gr.update()
                i = c.index(u) if u in c else 0
                return gr.update(value=c[(i + k) % len(c)])

            ds.change(on_ds, ds, clip)
            prev_b.click(lambda d, u: step(d, u, -1), [ds, clip], clip)
            next_b.click(lambda d, u: step(d, u, 1), [ds, clip], clip)
            for comp in (clip, show_all):
                comp.change(show_clip, [ds, clip, show_all], [video, out])
            demo.load(show_clip, [ds, clip, show_all], [video, out])

        with gr.Tab("Key findings"):
            figs = [(f, t) for f, t in CONFIG["findings"]
                    if os.path.exists(os.path.join(CONFIG["figures_dir"], f))]
            if figs:
                for f, text in figs:
                    gr.HTML(f'<p class="finding">{html.escape(text)}</p>')
                    gr.Image(os.path.join(CONFIG["figures_dir"], f), show_label=False,
                             interactive=False, height=430)
            else:
                gr.HTML('<p class="finding">1. With the same small amount of labels, '
                        'self-supervised models make far fewer mistakes.</p>')
                gr.Plot(fig_finding1())
                gr.HTML('<p class="finding">2. Supervised models only catch up with thousands of hours '
                        'of labelled video.</p>')
                gr.Plot(fig_finding2())
                gr.HTML('<p class="finding">3. Accuracy against speed.</p>')
                gr.Plot(fig_finding3())

        with gr.Tab("Live test"):
            gr.Markdown("Record or upload a short clip of **one sentence** (3–8 s, face clearly visible). "
                        "Each lip-reading module runs on the cloud GPU; the same video goes to every ticked model.")
            with gr.Row():
                pick = gr.CheckboxGroup(list(CONFIG["backends"]), value=list(CONFIG["backends"]),
                                        label="Models", scale=4)
                chk = gr.Button("Check servers", scale=1)
            srv_info = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=4):
                    vin = gr.Video(label="Your clip", sources=["upload", "webcam"], height=300)
                    ref_in = gr.Textbox(label="What was actually said (optional — marks the wrong words)",
                                        placeholder="e.g. the robot can help people in the hospital")
                    run_b = gr.Button("Read lips", variant="primary")
                with gr.Column(scale=6):
                    live_out = gr.HTML()
            chk.click(check_servers, None, srv_info)
            run_b.click(live_test, [vin, ref_in, pick], live_out)
    return demo


# ============================ 假数据（试运行用） ============================
def make_demo_data():
    sents = ["the robot can help people in the hospital",
             "we tested thirty models on the same videos",
             "lip reading works without any sound",
             "please turn left at the next corner",
             "this is a very quiet room",
             "self supervised learning needs fewer labels"]
    vocab = "the a to of in and people robot video help light quiet left right model sound".split()
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    models = pd.read_csv(CONFIG["models_csv"], dtype=str).fillna("")
    rnd = random.Random(0)
    for _, row in models.iterrows():
        if row["status"] == "pending":
            continue
        try:
            base = float(str(row["official_wer"]).split()[0]) / 100
        except ValueError:
            base = 0.3
        for ds, shift in (("bbc", 0.05), ("lrs3_300", 0.0)):
            recs = []
            for k, s in enumerate(sents):
                ws = s.split()
                hyp = [rnd.choice(vocab) if rnd.random() < min(base + shift, 0.95) else w for w in ws]
                recs.append({"utt_id": f"{ds}_{k:03d}", "ref": s, "hyp": " ".join(hyp),
                             "seconds": round(rnd.uniform(0.3, 4.0) * (2 if "huge" in row["size"] else 1), 2)})
            pd.DataFrame(recs).to_csv(os.path.join(CONFIG["results_dir"], f"{row['id']}_{ds}.csv"),
                                      index=False)
    print("Demo results written to", CONFIG["results_dir"])


# ============================ 自动挑选演示片段 ============================
def suggest_clips(top=3):
    """按 3 类找演示片段：全对 / 自监督对但监督错 / 全错"""
    for ds_label, ds in CONFIG["datasets"].items():
        mids = [m for m in CONFIG["default_models"] if (m, ds) in RESULTS]
        if not mids:
            continue
        stats = []
        for utt in clip_choices(ds_label):
            w = {}
            for m in mids:
                df = RESULTS[(m, ds)]
                if utt in df.index:
                    e, n = errors_and_len(df.loc[utt, "ref"], df.loc[utt, "hyp"])
                    w[m] = e / n
            sup = [v for m, v in w.items() if MODELS.loc[m, "group"] == "supervised"]
            ssl = [v for m, v in w.items() if MODELS.loc[m, "group"] != "supervised"]
            if not sup or not ssl:
                continue
            stats.append((utt, sum(w.values()) / len(w), min(ssl), min(sup)))
        if not stats:
            continue
        print(f"\n===== {ds_label} =====")
        easy = sorted(stats, key=lambda x: x[1])[:top]
        gap = sorted(stats, key=lambda x: x[2] - x[3])[:top]
        hard = sorted(stats, key=lambda x: -x[1])[:top]
        print("A. 大部分模型都认对（开场用）:", [u for u, *_ in easy])
        print("B. 自监督认对、监督认错（重点）:",
              [f"{u} (自监督最好 {a:.0%} / 监督最好 {b:.0%})" for u, _, a, b in gap])
        print("C. 所有模型都认错（讲局限性）:", [u for u, *_ in hard])


# ============================ 入口 ============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="生成假数据并运行")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="生成公网链接")
    ap.add_argument("--big", action="store_true", help="录制模式：字体放大")
    ap.add_argument("--backend", default="", help="唇语识别服务地址，默认 http://127.0.0.1:7001")
    ap.add_argument("--suggest", action="store_true", help="只打印推荐的演示片段，不启动界面")
    args = ap.parse_args()
    if args.demo:
        CONFIG["results_dir"] = os.path.join(HERE, "results_demo")
        CONFIG["figures_dir"] = os.path.join(HERE, "no_figures")
        if not os.path.exists(CONFIG["models_csv"]):
            CONFIG["models_csv"] = os.path.join(os.path.dirname(HERE), "models.csv")
        make_demo_data()
    if args.backend:
        CONFIG["backend_url"] = args.backend
    load_all()
    print(f"数据文件夹: {DATA}")
    print(f"  结果 {len(RESULTS)} 个，视频文件夹存在: {os.path.isdir(CONFIG['videos_dir'])}，"
          f"图片文件夹存在: {os.path.isdir(CONFIG['figures_dir'])}")
    if args.suggest:
        suggest_clips()
        raise SystemExit
    if args.big:
        CSS = CSS + CSS_BIG
    ui = build_ui()
    if GR_MAJOR >= 6:         # Gradio 6：theme / css 放在 launch()
        ui.launch(server_port=args.port, share=args.share, theme=make_theme(), css=CSS,
                  allowed_paths=[DATA, CONFIG["videos_dir"], CONFIG["figures_dir"]])
    else:                     # Gradio 4 / 5：已在 Blocks() 里设置
        ui.launch(server_port=args.port, share=args.share,
                  allowed_paths=[DATA, CONFIG["videos_dir"], CONFIG["figures_dir"]])
