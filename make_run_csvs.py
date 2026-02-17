#!/usr/bin/env python3
import csv, json, sys, math, re
from pathlib import Path
from collections import defaultdict

num_re = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")

def to_num(x):
    if x is None: return None
    if isinstance(x, bool): return 1.0 if x else 0.0
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        s = x.strip().rstrip(",")
        if num_re.match(s):
            try: return float(s)
            except: return None
    return None

def pctl(xs, p):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    xs = sorted(xs)
    k = (len(xs)-1) * (p/100.0)
    f = int(math.floor(k))
    c = min(f+1, len(xs)-1)
    if f == c: return float(xs[f])
    return float(xs[f] + (xs[c]-xs[f]) * (k-f))

def flatten(prefix, obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flatten(key, v, out)
    elif isinstance(obj, list):
        return
    else:
        val = to_num(obj)
        if val is not None:
            out[prefix] = val

def pick_main_metric(task_name: str, flat: dict):
    t = task_name.lower()
    if "truthful" in t:
        prefs = ["mc2", "acc_norm", "acc"]
    elif "gsm8k" in t:
        prefs = ["exact_match", "exact_match_flexible", "acc", "acc_norm"]
    elif "hellaswag" in t or "arc" in t or "winogrande" in t or "wino" in t:
        prefs = ["acc_norm", "acc"]
    else:
        prefs = ["acc_norm","acc","exact_match","mc2"]

    for p in prefs:
        if p in flat:
            return p, flat[p]
    for p in prefs:
        for k, v in flat.items():
            if k.endswith("." + p):
                return k, v
    return None, None

def extract_tasks_from_results_files(run_dir: Path):
    best = {}  # task -> (mtime, metrics_dict)
    for fp in run_dir.rglob("results_*.json"):
        try:
            j = json.loads(fp.read_text())
        except Exception:
            continue
        results = j.get("results")
        if not isinstance(results, dict) or not results:
            continue
        mtime = fp.stat().st_mtime
        for task_name, metrics in results.items():
            if task_name not in best or mtime > best[task_name][0]:
                best[task_name] = (mtime, metrics)
    return {k: v[1] for k, v in best.items()}

def process_accuracy(run_dir: Path, run_id: str):
    tasks = extract_tasks_from_results_files(run_dir)
    acc_rows = []
    main_row = {"run_id": run_id}

    for task_name, metrics in sorted(tasks.items()):
        flat = {}
        flatten("", metrics, flat)

        for metric_name, val in flat.items():
            acc_rows.append({"run_id": run_id, "task": task_name, "metric": metric_name, "value": val})

        mk, mv = pick_main_metric(task_name, flat)
        if mv is not None:
            main_row[f"{task_name}:{mk}"] = mv

    out_long = run_dir / "accuracy_metrics.csv"
    with out_long.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id","task","metric","value"])
        w.writeheader()
        for r in acc_rows:
            w.writerow(r)

    out_wide = run_dir / "accuracy_main.csv"
    cols = ["run_id"] + sorted([k for k in main_row.keys() if k != "run_id"])
    with out_wide.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(main_row)

def process_perf(run_dir: Path, run_id: str):
    perf_path = run_dir / "perf.json"
    if not perf_path.exists():
        return
    perf = json.loads(perf_path.read_text())
    rid = perf.get("run_id", run_id)

    tps = perf.get("throughput_tok_s") or {}
    lat = perf.get("latency_s") or {}
    suc = perf.get("success") or {}
    probe = perf.get("probe") or {}
    usage = perf.get("usage_tokens") or {}

    # Minimal (tok/s only)
    out_min = run_dir / "perf_tok_s.csv"
    row_min = {
        "run_id": rid,
        "prompt_tok_s": tps.get("prompt_tok_s"),
        "completion_tok_s": tps.get("completion_tok_s"),
        "total_tok_s": tps.get("total_tok_s"),
    }
    with out_min.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_min.keys()))
        w.writeheader()
        w.writerow(row_min)

    # Full
    out_full = run_dir / "perf_full.csv"
    row_full = {
        "run_id": rid,
        "endpoint": perf.get("endpoint"),
        "mode": perf.get("mode"),
        "model": perf.get("model"),
        "probe_n_requests": probe.get("n_requests"),
        "probe_concurrency": probe.get("concurrency"),
        "probe_max_tokens": probe.get("max_tokens"),
        "probe_timeout_s": probe.get("timeout_s"),
        "ok": suc.get("ok"),
        "failed": suc.get("failed"),
        "lat_p50_s": lat.get("p50"),
        "lat_p95_s": lat.get("p95"),
        "lat_p99_s": lat.get("p99"),
        "lat_mean_s": lat.get("mean"),
        "wall_time_s": perf.get("wall_time_s"),
        "sum_prompt_tokens": usage.get("sum_prompt_tokens"),
        "sum_completion_tokens": usage.get("sum_completion_tokens"),
        "sum_total_tokens": usage.get("sum_total_tokens"),
        "prompt_tok_s": tps.get("prompt_tok_s"),
        "completion_tok_s": tps.get("completion_tok_s"),
        "total_tok_s": tps.get("total_tok_s"),
    }
    with out_full.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_full.keys()))
        w.writeheader()
        w.writerow(row_full)

def process_memory_summary(run_dir: Path, run_id: str):
    # accept both memory-*.csv and memory*.csv
    mem_paths = list(run_dir.glob("memory*.csv"))
    if not mem_paths:
        return
    # prefer memory-<RUN_ID>.csv if present
    mem_path = None
    for p in mem_paths:
        if p.name == f"memory-{run_id}.csv":
            mem_path = p
            break
    if mem_path is None:
        mem_path = sorted(mem_paths)[0]

    by_gpu = defaultdict(lambda: {"util":[], "pwr":[], "temp":[], "mem":[], "mem_tot":[]})
    with mem_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gi = (row.get("gpu_index") or "").strip()
            if gi in ("END","ERROR","") or gi is None:
                continue
            try:
                g = int(gi)
            except:
                continue
            def n(k):
                try: return float(row.get(k,""))
                except: return None
            by_gpu[g]["util"].append(n("gpu_util_pct"))
            by_gpu[g]["pwr"].append(n("power_W"))
            by_gpu[g]["temp"].append(n("temp_C"))
            by_gpu[g]["mem"].append(n("mem_used_MiB"))
            by_gpu[g]["mem_tot"].append(n("mem_total_MiB"))

    out_mem = run_dir / "memory_summary.csv"
    fields = ["run_id","gpu_index","n_samples",
              "util_mean_pct","util_p95_pct","util_max_pct",
              "power_mean_W","power_max_W",
              "temp_mean_C","temp_max_C",
              "mem_used_mean_MiB","mem_used_max_MiB","mem_total_MiB",
              "source_file"]
    with out_mem.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in sorted(by_gpu.keys()):
            d = by_gpu[g]
            util = [x for x in d["util"] if x is not None]
            pwr  = [x for x in d["pwr"] if x is not None]
            temp = [x for x in d["temp"] if x is not None]
            mem  = [x for x in d["mem"] if x is not None]
            memt = [x for x in d["mem_tot"] if x is not None]
            w.writerow({
                "run_id": run_id,
                "gpu_index": g,
                "n_samples": len(util),
                "util_mean_pct": (sum(util)/len(util)) if util else None,
                "util_p95_pct": pctl(util, 95),
                "util_max_pct": max(util) if util else None,
                "power_mean_W": (sum(pwr)/len(pwr)) if pwr else None,
                "power_max_W": max(pwr) if pwr else None,
                "temp_mean_C": (sum(temp)/len(temp)) if temp else None,
                "temp_max_C": max(temp) if temp else None,
                "mem_used_mean_MiB": (sum(mem)/len(mem)) if mem else None,
                "mem_used_max_MiB": max(mem) if mem else None,
                "mem_total_MiB": (sum(memt)/len(memt)) if memt else None,
                "source_file": mem_path.name,
            })

def process_run(run_dir: Path):
    run_id = run_dir.name
    acc_path = run_dir / "accuracy.json"
    if acc_path.exists():
        try:
            j = json.loads(acc_path.read_text())
            run_id = j.get("run_id", run_id)
        except Exception:
            pass

    process_accuracy(run_dir, run_id)
    process_perf(run_dir, run_id)
    process_memory_summary(run_dir, run_id)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 make_run_csvs.py <run_dir_or_runs_dir>")
        sys.exit(2)

    p = Path(sys.argv[1]).resolve()
    if not p.exists():
        print(f"ERROR: not found: {p}")
        sys.exit(1)

    if p.name.startswith("QWEN-run-"):
        process_run(p)
        print(f"OK: {p}")
        return

    run_dirs = sorted([d for d in p.glob("QWEN-run-*") if d.is_dir()])
    for rd in run_dirs:
        process_run(rd)
        print(f"OK: {rd}")

if __name__ == "__main__":
    main()
