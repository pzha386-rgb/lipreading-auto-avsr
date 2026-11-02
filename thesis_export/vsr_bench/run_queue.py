# -*- coding: utf-8 -*-
"""
评测队列：读 models.csv，按 runner 类型自动调用评测脚本

  已完成的（结果文件存在）自动跳过 -> 中断后重新运行即可续跑
  某个模型出错不影响后面的模型
  每个任务一个日志：logs/{编号}_{数据集}.log

常用命令：
  python run_queue.py --dry-run --datasets bbc                 # 只打印命令，检查路径
  python run_queue.py --datasets bbc --limit 5                 # 所有模型冒烟测试（各 5 段）
  python run_queue.py --datasets bbc                           # 所有可运行模型跑 BBC
  python run_queue.py --datasets lrs3 --ids S0 S1 R1 B2        # 指定模型
  nohup python -u run_queue.py --datasets lrs3 > /root/autodl-tmp/logs/night1.log 2>&1 &
  python run_queue.py --summary                                # 汇总所有结果
"""
import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

from common import HERE, LOGS, RESULTS

PY = sys.executable

# 每种 runner 怎么调用（第 2 天再加 vsrml / usr2 / dllm）
RUNNERS = {
    "raven": "{py} -u {here}/eval_raven_family.py --id {id} --ckpt {ckpt} --dataset {ds} --beam {beam} {lm} --gpu-decode {limit}",
    "autoavsr": "{py} -u {here}/eval_autoavsr_cache.py --id {id} --ckpt {ckpt} --dataset {ds} {limit}",
    "vsrml": "{py} -u {here}/eval_vsrml.py --id {id} --model-dir {ckpt} --ini {arch} --dataset {ds} {lmdir} {limit}",
    "usr2": "{py} -u {here}/eval_usr2.py --id {id} --ckpt {ckpt} --backbone {arch} --dataset {ds} {dec} {limit}",
}


def build_cmd(row, ds, limit):
    runner = row["runner"]
    if runner not in RUNNERS:
        return None, f"runner '{runner}' 还没接入（第 2 天加）"
    if "FILL" in row["ckpt"] or (row["lm"] == "yes" and "FILL" in row["lm_path"]):
        return None, "models.csv 里还有 FILL_ 没填"
    if row["ckpt"].startswith("/") and not os.path.exists(row["ckpt"]):
        return None, f"权重不存在: {row['ckpt']}"
    lm = f"--lm {row['lm_path']}" if row["lm"] == "yes" else ""
    if lm and row.get("lm_weight", ""):
        lm += f" --lm-weight {row['lm_weight']}"
    beam = row["beam"] if row["beam"].isdigit() else "40"
    dec = ""
    if row["beam"].isdigit():
        dec += f" --beam {row['beam']}"
    try:
        dec += f" --ctc-weight {float(row['ctc_weight'])}"
    except ValueError:
        pass
    lmdir = f"--lm-dir {row['lm_path']}" if row["lm"] == "yes" else ""
    cmd = RUNNERS[runner].format(py=PY, here=HERE, id=row["id"], ckpt=row["ckpt"], ds=ds, arch=row["arch"],
                                 beam=beam, lm=lm, dec=dec, lmdir=lmdir, limit=f"--limit {limit}" if limit else "")
    return " ".join(cmd.split()), None


def out_path(mid, ds, limit):
    p = os.path.join(RESULTS, f"{mid}_{ds}.csv")
    return p.replace(".csv", f"_limit{limit}.csv") if limit else p


def run(a):
    df = pd.read_csv(os.path.join(HERE, "models.csv"), dtype=str).fillna("")
    if a.ids:
        df = df[df["id"].isin(a.ids)]
    else:
        df = df[df["status"].isin(a.status)]
    os.makedirs(LOGS, exist_ok=True)

    plan = []
    for ds in a.datasets:
        for _, row in df.iterrows():
            cmd, why = build_cmd(row, ds, a.limit)
            plan.append((row["id"], ds, cmd, why))

    print(f"{'编号':6s} {'数据集':10s} 状态")
    for mid, ds, cmd, why in plan:
        done = os.path.exists(out_path(mid, ds, a.limit))
        print(f"{mid:6s} {ds:10s} " + ("已完成，跳过" if done and not a.force else (why or "待运行")))
    if a.dry_run:
        print("\n---- 命令 ----")
        for mid, ds, cmd, why in plan:
            if cmd:
                print(cmd)
        return

    todo = [(m, d, c) for m, d, c, w in plan
            if c and not (os.path.exists(out_path(m, d, a.limit)) and not a.force)]
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))

    def one(job):
        mid, ds, cmd = job
        log = os.path.join(LOGS, f"{mid}_{ds}{'_limit' + str(a.limit) if a.limit else ''}.log")
        print(f">>> {time.strftime('%H:%M:%S')}  start {mid} on {ds}", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            code = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)
        mins = (time.time() - t0) / 60
        ok = code == 0 and os.path.exists(out_path(mid, ds, a.limit))
        wer = ""
        if ok:
            meta = json.load(open(out_path(mid, ds, a.limit).replace(".csv", ".meta.json")))
            wer = f"WER {meta['wer']:.2f}%  failed {meta['n_failed']}"
        print(f"<<< {time.strftime('%H:%M:%S')}  {mid} on {ds}: {'OK' if ok else 'FAILED (看日志 ' + log + ')'}"
              f"  {mins:.1f} min  {wer}", flush=True)
        return (mid, ds, "OK" if ok else "FAILED", round(mins, 1), wer)

    from concurrent.futures import ThreadPoolExecutor
    print(f"\n并行 {a.jobs} 个任务，每个 {a.threads} 个 CPU 线程，共 {len(todo)} 个任务\n", flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        log_rows = list(ex.map(one, todo))

    print("\n========== 本次运行 ==========")
    for r in log_rows:
        print(*r)


def summary():
    df = pd.read_csv(os.path.join(HERE, "models.csv"), dtype=str).fillna("").set_index("id")
    rows = []
    if not os.path.isdir(RESULTS):
        print("还没有结果文件夹:", RESULTS)
        return
    for fn in sorted(os.listdir(RESULTS)):
        if not fn.endswith(".meta.json") or "_limit" in fn:
            continue
        m = json.load(open(os.path.join(RESULTS, fn)))
        mid = m["model_id"]
        info = df.loc[mid] if mid in df.index else {}
        rows.append({"id": mid, "dataset": m["dataset"], "type": info.get("type", ""),
                     "family": info.get("family", ""), "official_wer": info.get("official_wer", ""),
                     "wer": m["wer"], "clips": m["n_clips"], "failed": m["n_failed"],
                     "sec_per_clip": m.get("mean_seconds"), "params_M": m.get("params_M"),
                     "gpu_MB": m.get("peak_gpu_MB")})
    if not rows:
        print("还没有结果")
        return
    s = pd.DataFrame(rows)
    wide = s.pivot_table(index=["id", "type", "family", "official_wer"], columns="dataset",
                         values="wer").reset_index()
    out = os.path.join(RESULTS, "summary.csv")
    s.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(wide.sort_values(wide.columns[-1]).to_string(index=False))
    print(f"\n详细汇总已保存: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["bbc"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--status", nargs="*", default=["new", "ref"], help="默认只跑 new 和 ref")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="已完成的也重跑")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="同时运行几个模型")
    ap.add_argument("--threads", type=int, default=2, help="每个模型用几个 CPU 线程")
    a = ap.parse_args()
    summary() if a.summary else run(a)
