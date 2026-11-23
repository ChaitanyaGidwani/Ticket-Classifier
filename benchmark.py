"""Benchmark: run N tickets through the classifier, record timing + outputs.
Usage (from your project folder, venv active):
    python benchmark.py                      # uses tickets_100.csv
    python benchmark.py my_tickets.csv
Outputs: benchmark_results.csv (per-ticket), benchmark_summary.json / .md
"""
import json, platform, sys, time
import numpy as np
import pandas as pd
import torch
from classifier import TicketClassifier

path = sys.argv[1] if len(sys.argv) > 1 else "tickets_100.csv"
df = pd.read_csv(path)
texts = df["text"].astype(str).tolist()
n = len(texts)

t0 = time.perf_counter()
clf = TicketClassifier()
load_s = time.perf_counter() - t0
device = str(next(clf.model.parameters()).device)
clf.classify("warm up", check_duplicates=False)          # exclude one-off warm-up

# 1) per-ticket latency (end-to-end classify, one ticket at a time)
lat, results = [], []
t_all = time.perf_counter()
for t in texts:
    t1 = time.perf_counter()
    results.append(clf.classify(t, check_duplicates=False))
    lat.append((time.perf_counter() - t1) * 1000)
total_s = time.perf_counter() - t_all

# 2) pure embedding throughput (all tickets in batches)
t1 = time.perf_counter(); clf._enc(texts); enc_s = time.perf_counter() - t1

# 3) duplicate detection over the whole batch
t1 = time.perf_counter(); pairs = clf.find_duplicate_pairs(texts); dup_s = time.perf_counter() - t1

out = pd.DataFrame({
    "id": df.get("id", range(1, n + 1)), "text": texts,
    "pred_primary": [r["primary"] for r in results],
    "pred_labels": ["|".join(r["labels"]) for r in results],
    "top_score": [round(r["scores"][r["primary"]], 3) for r in results],
    "margin": [round(r["margin"], 3) for r in results],
    "pred_urgent": [r["urgent"] for r in results],
    "urgent_reason": [r["urgent_reason"] for r in results],
    "uncertain": [r["uncertain"] for r in results],
    "uncertain_reason": [r["uncertain_reason"] for r in results],
    "latency_ms": [round(x, 2) for x in lat],
})

summary = {
    "model": clf.model_name, "device": device, "torch": torch.__version__,
    "machine": platform.platform(), "tickets": n,
    "model_load_s": round(load_s, 2),
    "total_classify_s": round(total_s, 3),
    "avg_ms": round(float(np.mean(lat)), 2), "p50_ms": round(float(np.percentile(lat, 50)), 2),
    "p95_ms": round(float(np.percentile(lat, 95)), 2), "max_ms": round(float(np.max(lat)), 2),
    "throughput_tickets_per_s": round(n / total_s, 1),
    "batch_embed_s": round(enc_s, 3), "batch_duplicate_scan_s": round(dup_s, 3),
    "uncertain_flagged": int(out["uncertain"].sum()), "urgent_flagged": int(out["pred_urgent"].sum()),
    "duplicate_pairs_found": len(pairs),
}

# Accuracy (only if ground-truth columns exist)
if "true_labels" in df:
    tl = df["true_labels"].fillna("").astype(str).tolist()
    lab = [i for i, s in enumerate(tl) if s]
    truth = [set(tl[i].split("|")) for i in lab]
    pred = [set(results[i]["labels"]) for i in lab]
    summary["labeled_tickets"] = len(lab)
    summary["primary_label_acc"] = round(float(np.mean([results[i]["primary"] in tl[i].split("|") for i in lab])), 3)
    summary["exact_set_match"] = round(float(np.mean([a == b for a, b in zip(truth, pred)])), 3)
    tp = sum(len(a & b) for a, b in zip(truth, pred)); fp = sum(len(b - a) for a, b in zip(truth, pred)); fn = sum(len(a - b) for a, b in zip(truth, pred))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    summary["label_precision"], summary["label_recall"] = round(p, 3), round(r, 3)
    summary["label_f1"] = round(2 * p * r / max(p + r, 1e-9), 3)
    out["true_labels"] = tl
    out["correct"] = [(set(tl[i].split("|")) == set(results[i]["labels"])) if tl[i] else None for i in range(n)]
    if "true_urgent" in df:
        tu = df["true_urgent"].astype(bool).tolist(); pu = out["pred_urgent"].tolist()
        tpu = sum(a and b for a, b in zip(tu, pu)); fpu = sum((not a) and b for a, b in zip(tu, pu)); fnu = sum(a and (not b) for a, b in zip(tu, pu))
        summary["urgent_precision"] = round(tpu / max(tpu + fpu, 1), 3)
        summary["urgent_recall"] = round(tpu / max(tpu + fnu, 1), 3)
    if "note" in df:
        note = df["note"].fillna("").astype(str)
        vague = [i for i in range(n) if note[i] == "vague"]
        if vague:
            summary["vague_flagged_uncertain"] = f"{sum(results[i]['uncertain'] for i in vague)}/{len(vague)}"
        exp = []
        ids = df["id"].tolist() if "id" in df else list(range(1, n + 1))
        pos = {v: i for i, v in enumerate(ids)}
        for i in range(n):
            if note[i].startswith("dup_of:"):
                exp.append(tuple(sorted((i, pos[int(note[i].split(":")[1])]))))
        found = {tuple(sorted((a, b))) for a, b, _ in pairs}
        if exp:
            summary["expected_duplicates_found"] = f"{sum(e in found for e in exp)}/{len(exp)}"

out.to_csv("benchmark_results.csv", index=False)
json.dump(summary, open("benchmark_summary.json", "w"), indent=2)
md = "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in summary.items())
open("benchmark_summary.md", "w").write(md)
print(md)
if "correct" in out:
    bad = out[out["correct"] == False]
    print(f"\nMisclassified ({len(bad)}):")
    print(bad[["text", "true_labels", "pred_labels", "top_score"]].to_string(index=False))
print("\nSaved: benchmark_results.csv, benchmark_summary.json, benchmark_summary.md")
