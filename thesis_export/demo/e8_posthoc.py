"""E8 Full-System post-hoc: coverage sweep + 2-system ROVER + waterfall + curve."""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import jiwer

T1_CSV  = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV  = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"
OUT_DIR = "/root/autodl-tmp/demo/e8_results"
os.makedirs(OUT_DIR, exist_ok=True)
COVERAGES = [1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]

ID, PRED, REF, CONF = "id", "hyp", "ref", "confA_logp"

def wer_of(df):
    return 100.0 * jiwer.wer([str(x) for x in df[REF]], [str(x) for x in df[PRED]])

def sweep(df):
    d = df.sort_values(CONF, ascending=False).reset_index(drop=True); N=len(d)
    rows=[]
    for c in COVERAGES:
        k = max(1, int(round(c*N))); rows.append({"coverage":c,"n":k,"wer":wer_of(d.iloc[:k])})
    return pd.DataFrame(rows)

def rover2(a, b):
    m = a[[ID,REF,PRED,CONF]].merge(b[[ID,PRED,CONF]], on=ID, suffixes=("_a","_b"))
    pick = m[f"{CONF}_a"] >= m[f"{CONF}_b"]
    m[PRED] = np.where(pick, m[f"{PRED}_a"], m[f"{PRED}_b"])
    m[CONF] = np.where(pick, m[f"{CONF}_a"], m[f"{CONF}_b"])
    return m[[ID,REF,PRED,CONF]]

def main():
    results = {}
    df_t1 = pd.read_csv(T1_CSV); print(f"T1 baseline: {wer_of(df_t1):.2f}% ({len(df_t1)} clips)")
    results["T1"] = sweep(df_t1); print(results["T1"].to_string(index=False))

    df_b2 = pd.read_csv(B2_CSV); print(f"\nB2 baseline: {wer_of(df_b2):.2f}% ({len(df_b2)} clips)")
    results["B2"] = sweep(df_b2); print(results["B2"].to_string(index=False))

    df_rv = rover2(df_t1, df_b2)
    print(f"\nROVER baseline: {wer_of(df_rv):.2f}% ({len(df_rv)} clips)")
    results["ROVER"] = sweep(df_rv); print(results["ROVER"].to_string(index=False))

    rows=[]
    for n, sw in results.items():
        base = sw.loc[sw.coverage==1.00,"wer"].values[0]
        at70 = sw.loc[sw.coverage==0.70,"wer"].values[0]
        at50 = sw.loc[sw.coverage==0.50,"wer"].values[0]
        rows.append({"system":n, "baseline":f"{base:.2f}%", "keep=70%":f"{at70:.2f}%",
                     "keep=50%":f"{at50:.2f}%", "drop@50 (pp)":f"{base-at50:+.2f}"})
    wf = pd.DataFrame(rows); wf.to_csv(f"{OUT_DIR}/waterfall.csv", index=False)
    print("\n=== WATERFALL ==="); print(wf.to_string(index=False))

    plt.figure(figsize=(7.5,5))
    colors={"T1":"#ef4444","B2":"#3b82f6","ROVER":"#22c55e"}
    for n, sw in results.items():
        plt.plot(sw.coverage*100, sw.wer, "o-", lw=2, color=colors.get(n,"k"), label=n)
    plt.gca().invert_xaxis()
    plt.xlabel("Coverage (% kept, highest confidence first)")
    plt.ylabel("WER (%)")
    plt.title("Multi-speaker test — WER vs coverage")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/wer_vs_coverage.png", dpi=150)
    for n, sw in results.items(): sw.to_csv(f"{OUT_DIR}/sweep_{n}.csv", index=False)
    print(f"\n✓ All outputs in {OUT_DIR}")

if __name__ == "__main__": main()
