# -*- coding: utf-8 -*-
"""
生成论文 / PPT 用的核心图表（英文标签）
输入：results_v2/summary_main.csv, summary_robustness.csv, summary_speed.csv
输出：/root/autodl-tmp/figures/fig*.png（300 dpi）和 table_main.csv

运行：python make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

R = f"{ROOT}/results_v2"
OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)

main = pd.read_csv(f"{R}/summary_main.csv").set_index("id")
rob = pd.read_csv(f"{R}/summary_robustness.csv").set_index("id")
speed = pd.read_csv(f"{R}/summary_speed.csv").set_index("id")

BLUE, GREEN, PURPLE, GREY = "#2F5D8A", "#1F8A70", "#7A4E9E", "#9AA5B1"


def group(t):
    t = str(t)
    if t.startswith("supervised"):
        return "Supervised"
    if "semi" in t:
        return "Self-sup. + semi-sup."
    return "Self-supervised"


COLOR = {"Supervised": BLUE, "Self-supervised": GREEN, "Self-sup. + semi-sup.": PURPLE}
main["group"] = main["type"].map(group)

NAME = {
    "S6": "Auto-AVSR 23h", "S0": "Auto-AVSR 438h", "S1": "Auto-AVSR 1759h", "T1": "Auto-AVSR 3291h",
    "S2": "Auto-AVSR (VSR-ML)", "S2lm": "Auto-AVSR (VSR-ML) +LM",
    "S4": "Ma2022 LRS3", "S4lm": "Ma2022 LRS3 +LM", "S5": "Ma2022 LRS2", "S5lm": "Ma2022 LRS2 +LM",
    "T3": "BRAVEn low +ST", "B1": "BRAVEn low", "B2": "BRAVEn low +ST +LM",
    "B3": "BRAVEn high", "B4": "BRAVEn high +ST", "B5": "BRAVEn high +ST +LM",
    "T4": "RAVEn low +ST", "R1": "RAVEn low +ST +LM", "R2": "RAVEn high +ST", "R3": "RAVEn high +ST +LM",
    "U1": "USR2 Large low", "U2": "USR2 Base+ high", "U3": "USR2 Large high", "U4": "USR2 Huge high",
    "U1g": "USR2 Large low (greedy)", "U2g": "USR2 Base+ (greedy)", "U3g": "USR2 Large high (greedy)",
    "U4g": "USR2 Huge (greedy)", "U4b10": "USR2 Huge (beam 10)",
}
name = lambda i: NAME.get(i, i)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def save(fig, fn):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fn}", dpi=300)
    plt.close(fig)
    print("saved", fn)


# ---------- Fig 1：标签数量（核心图）----------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sup = [("S6", 23), ("S0", 438), ("S1", 1759), ("T1", 3291)]
sup = [(i, h) for i, h in sup if i in main.index]
ax.plot([h for _, h in sup], [main.loc[i, "lrs3_300"] for i, _ in sup], "-o", color=BLUE, lw=2, ms=8,
        label="Supervised (Auto-AVSR)")
for i, h in sup:
    ax.annotate(f"{main.loc[i, 'lrs3_300']:.0f}%", (h, main.loc[i, "lrs3_300"]), xytext=(6, 4),
                textcoords="offset points", fontsize=9, color=BLUE)
ssl = {30: ["T3", "B1", "T4", "U1"], 433: ["B4", "B3", "R2", "U3", "U4"]}
first = True
for h, ids in ssl.items():
    ids = sorted([i for i in ids if i in main.index], key=lambda i: main.loc[i, "lrs3_300"])
    ly = -1e9
    for i in ids:
        yv = main.loc[i, "lrs3_300"]
        c = COLOR[main.loc[i, "group"]]
        ax.scatter(h, yv, s=60, color=c, zorder=3,
                   label="Self-supervised (labelled hours = fine-tuning)" if first else None)
        first = False
        ly = max(yv, ly + 2.2)               # 标签至少间隔 2.2，避免重叠
        ax.annotate(f"{name(i)} ({yv:.0f}%)", (h, yv), xytext=(h * 1.12, ly), fontsize=7, va="center",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
ax.set_xscale("log")
ax.set_xlabel("Hours of human/auto-labelled video used for training (log)")
ax.set_ylabel("WER on LRS3 (300-clip subset, %)")
ax.set_title("Self-supervised pre-training needs far fewer labels")
ax.legend(frameon=False, fontsize=9)
style(ax)
save(fig, "fig1_label_efficiency.png")

# ---------- Fig 2：总体排名 ----------
d = main.sort_values("lrs3_300")
fig, ax = plt.subplots(figsize=(8, 9))
y = np.arange(len(d))
ax.barh(y, d["lrs3_300"], color=[COLOR[g] for g in d["group"]], height=0.6, label="LRS3 (300)")
ax.scatter(d["bbc"], y, color="black", marker="|", s=120, zorder=3, label="BBC (out-of-domain)")
ax.set_yticks(y)
ax.set_yticklabels([name(i) for i in d.index], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("WER (%)  — lower is better")
ax.set_title("All configurations: LRS3 (bars) vs BBC (black ticks)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles + [ax.collections[0]], list(COLOR) + ["BBC"], frameon=False, fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
save(fig, "fig2_overall_ranking.png")

# ---------- Fig 3：域内 vs 域外 ----------
d = main[~main.index.str.endswith(("g", "b10"))]
fig, ax = plt.subplots(figsize=(6.5, 6))
for g, c in COLOR.items():
    s = d[d.group == g]
    ax.scatter(s["lrs3_300"], s["bbc"], color=c, s=50, label=g, zorder=3)
for i in ["U4", "T1", "T3", "B4", "S5", "S4", "U3"]:
    if i in d.index:
        ax.annotate(name(i), (d.loc[i, "lrs3_300"], d.loc[i, "bbc"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=8)
lim = [15, 70]
ax.plot(lim, lim, "--", color=GREY, lw=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("WER on LRS3 (in-domain, %)")
ax.set_ylabel("WER on BBC (out-of-domain, %)")
ax.set_title("The best model on LRS3 is not the best on BBC")
ax.legend(frameon=False, fontsize=8)
style(ax)
save(fig, "fig3_in_vs_out_domain.png")

# ---------- Fig 4：Self-training 和语言模型的作用 ----------
pairs = [("Self-training", "B1", "T3"), ("Self-training", "B3", "B4"),
         ("Language model", "T3", "B2"), ("Language model", "B4", "B5"),
         ("Language model", "T4", "R1"), ("Language model", "R2", "R3"),
         ("Language model", "S2", "S2lm"), ("Language model", "S4", "S4lm")]
pairs = [p for p in pairs if p[1] in main.index and p[2] in main.index]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, ds, title in zip(axes, ["lrs3_300", "bbc"], ["LRS3 (in-domain)", "BBC (out-of-domain)"]):
    delta = [main.loc[b, ds] - main.loc[a, ds] for _, a, b in pairs]
    cols = [GREEN if k == "Self-training" else PURPLE for k, _, _ in pairs]
    ax.barh(range(len(pairs)), delta, color=cols)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("ΔWER (points; negative = better)")
    ax.grid(axis="x", alpha=0.25)
axes[0].set_yticks(range(len(pairs)))
axes[0].set_yticklabels([f"{name(a)}  + {'ST' if k == 'Self-training' else 'LM'}" for k, a, b in pairs], fontsize=8)
axes[0].invert_yaxis()
fig.suptitle("Self-training (green) helps in both domains; LM gains (purple) are significant only in-domain")
save(fig, "fig4_st_and_lm.png")

# ---------- Fig 5：速度 vs 准确率 ----------
d = main.join(speed[["sec_per_clip", "RTF", "peak_gpu_MB"]], how="inner")
d = d[d["lrs3_300"] < 70]                      # S6（94%）太高，放在图注里说明
fig, ax = plt.subplots(figsize=(7.5, 5))
for g, c in COLOR.items():
    s = d[d.group == g]
    ax.scatter(s["sec_per_clip"], s["lrs3_300"], s=s["peak_gpu_MB"] / 60, color=c, alpha=0.75,
               label=g, zorder=3, edgecolor="white")
OFF = {"U4g": (6, 4), "U4": (-20, 10), "T1": (-95, -12), "T3": (8, 8), "B5": (6, -12), "S4": (6, 4), "U2g": (6, 4)}
for i, off in OFF.items():
    if i in d.index:
        ax.annotate(name(i), (d.loc[i, "sec_per_clip"], d.loc[i, "lrs3_300"]), xytext=off,
                    textcoords="offset points", fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Seconds per clip on RTX 5090 (log)  — lower is faster")
ax.set_ylabel("WER on LRS3 (300, %)")
ax.set_title("Accuracy vs speed (bubble size = GPU memory; Auto-AVSR 23h at 94% not shown)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
style(ax)
save(fig, "fig5_speed_vs_accuracy.png")

# ---------- Fig 6：鲁棒性 ----------
conds = [c for c in ["bbc", "bbc_blur2", "bbc_blur4", "bbc_low50", "bbc_low25",
                     "bbc_dark05", "bbc_dark03", "bbc_drop25", "bbc_drop50"] if c in rob.columns]
labels = {"bbc": "clean", "bbc_blur2": "blur σ2", "bbc_blur4": "blur σ4", "bbc_low50": "res 50%",
          "bbc_low25": "res 25%", "bbc_dark05": "light 0.5", "bbc_dark03": "light 0.3",
          "bbc_drop25": "drop 25%", "bbc_drop50": "drop 50%"}
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(conds))
for i in rob.index:
    c = COLOR.get(main.loc[i, "group"], GREY) if i in main.index else GREY
    ax.plot(x, rob.loc[i, conds], "-o", ms=5, lw=1.8, color=c, label=name(i), alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([labels[c] for c in conds], rotation=20)
ax.set_ylabel("WER on BBC (%)")
ax.set_title("Robustness: frame drops hurt every model; darkness barely matters")
ax.legend(frameon=False, fontsize=8, ncol=2)
style(ax)
save(fig, "fig6_robustness.png")

# ---------- Fig 7：解码方式（beam vs greedy）----------
pairs = [("U1", "U1g"), ("U2", "U2g"), ("U3", "U3g"), ("U4", "U4g")]
pairs = [p for p in pairs if all(q in main.index and q in speed.index for q in p)]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
x = np.arange(len(pairs))
for ax, col, src, yl in [(axes[0], "lrs3_300", main, "WER on LRS3 (300, %)"),
                         (axes[1], "sec_per_clip", speed, "Seconds per clip")]:
    ax.bar(x - 0.2, [src.loc[a, col] for a, _ in pairs], 0.4, color=PURPLE, label="beam 40")
    ax.bar(x + 0.2, [src.loc[b, col] for _, b in pairs], 0.4, color="#C9B3DD", label="greedy (beam 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([name(a).replace("USR2 ", "") for a, _ in pairs], fontsize=8)
    ax.set_ylabel(yl)
    style(ax)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Greedy decoding: up to ~18× faster for a small accuracy cost")
save(fig, "fig7_beam_vs_greedy.png")

# ---------- 论文主表 ----------
t = main.join(speed[["sec_per_clip", "RTF", "peak_gpu_MB"]], how="left")
t.insert(0, "model", [name(i) for i in t.index])
t = t[["model", "group", "official_wer", "lrs3", "lrs3_300", "bbc", "params_M", "sec_per_clip", "RTF", "peak_gpu_MB"]]
t.sort_values("lrs3_300").to_csv(f"{OUT}/table_main.csv")
print("saved table_main.csv")
print(f"\n全部输出在 {OUT}")
