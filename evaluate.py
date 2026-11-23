"""Expanded performance evaluation suite for Support Ticket Classifier.

Evaluates:
  1. Multi-label classification (Exact match, Primary accuracy, Hamming loss, Jaccard similarity, Macro/Micro/Weighted F1)
  2. Granular single-issue vs multi-issue performance breakdown
  3. Per-class precision, recall, F1, and support
  4. Urgency detection confusion matrix, sensitivity, specificity, and trigger source analysis
  5. Semantic duplicate detection precision, recall, F1, and similarity margins
  6. Confidence calibration & uncertainty triage (routing for human review)
  7. Inference latency and throughput benchmarks (mean, median, p95, p99, rps)
  8. Misclassification error taxonomy diagnostics
"""
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import MultiLabelBinarizer

from classifier import HUGGINGFACE_MODEL_NAME, LABELS, TicketClassifier
from test_data import DUP_PAIRS, TEST_SET


def run_eval(clf: TicketClassifier, verbose: bool = False) -> Dict[str, Any]:
    """Run comprehensive performance evaluation across all benchmark dimensions."""
    preds: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    # 1. Per-ticket evaluation & inference latency measurement
    for t in TEST_SET:
        t0 = time.perf_counter()
        r = clf.classify(t["text"], check_duplicates=False)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)
        preds.append(r)

        true_set = set(t["labels"])
        pred_set = set(r["labels"])
        is_exact = true_set == pred_set
        primary_correct = r["primary"] in true_set

        # Error categorization
        error_type = "None"
        if not is_exact:
            if not primary_correct:
                error_type = "Wrong Primary Label"
            elif pred_set < true_set:
                error_type = "Missed Label (Under-prediction)"
            elif pred_set > true_set:
                error_type = "Extra Label (Over-prediction)"
            else:
                error_type = "Mixed Mismatch"

        rows.append({
            "text": t["text"],
            "true": ", ".join(t["labels"]),
            "pred": ", ".join(r["labels"]),
            "correct": is_exact,
            "primary_correct": primary_correct,
            "error_type": error_type,
            "true_urgent": t["urgent"],
            "pred_urgent": r["urgent"],
            "urgent_match": t["urgent"] == r["urgent"],
            "uncertain": r["uncertain"],
            "uncertain_reason": r["uncertain_reason"],
            "primary_score": round(r["scores"][r["primary"]], 3),
            "margin": round(r["margin"], 3),
            "latency_ms": round(lat_ms, 2),
            "is_multi_issue": len(t["labels"]) > 1,
        })

    rows_df = pd.DataFrame(rows)

    # 2. Multi-Label Classification Metrics
    mlb = MultiLabelBinarizer(classes=LABELS)
    yt = mlb.fit_transform([t["labels"] for t in TEST_SET])
    yp = mlb.transform([r["labels"] for r in preds])

    report_dict = classification_report(
        yt, yp, target_names=LABELS, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T.round(3)

    exact_match = float(sum(r["correct"] for r in rows) / len(TEST_SET))
    primary_acc = float(sum(r["primary_correct"] for r in rows) / len(TEST_SET))
    hl = float(hamming_loss(yt, yp))
    jaccard_samples = float(jaccard_score(yt, yp, average="samples", zero_division=0))

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        yt, yp, average="weighted", zero_division=0
    )

    # 3. Single-Issue vs Multi-Issue Segmented Analysis
    single_mask = ~rows_df["is_multi_issue"]
    multi_mask = rows_df["is_multi_issue"]
    n_single = int(single_mask.sum())
    n_multi = int(multi_mask.sum())

    single_exact = float(rows_df.loc[single_mask, "correct"].mean()) if n_single > 0 else 0.0
    single_primary = float(rows_df.loc[single_mask, "primary_correct"].mean()) if n_single > 0 else 0.0

    multi_exact = float(rows_df.loc[multi_mask, "correct"].mean()) if n_multi > 0 else 0.0
    multi_primary = float(rows_df.loc[multi_mask, "primary_correct"].mean()) if n_multi > 0 else 0.0
    multi_partial = float(sum(
        len(set(t["labels"]) & set(r["labels"])) > 0
        for t, r in zip(TEST_SET, preds) if len(t["labels"]) > 1
    ) / max(n_multi, 1))

    # 4. Urgency Detection Metrics & Confusion Matrix
    true_urgents = [t["urgent"] for t in TEST_SET]
    pred_urgents = [r["urgent"] for r in preds]

    up, ur, uf, _ = precision_recall_fscore_support(
        true_urgents, pred_urgents, average="binary", zero_division=0
    )
    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(true_urgents, pred_urgents, labels=[False, True])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = (ur + specificity) / 2.0

    # Urgency trigger breakdown
    kw_count = sum(1 for r in preds if r["urgent"] and "keyword" in r["urgent_reason"])
    sem_count = sum(1 for r in preds if r["urgent"] and "semantic" in r["urgent_reason"])

    # 5. Semantic Duplicate Detection Evaluation
    dup_results = []
    dup_sims, non_dup_sims = [], []
    for a, b, is_dup in DUP_PAIRS:
        sim = clf.similarity(a, b)
        pred_dup = sim >= clf.dup_threshold
        if is_dup:
            dup_sims.append(sim)
        else:
            non_dup_sims.append(sim)
        dup_results.append({
            "ticket_a": a,
            "ticket_b": b,
            "true_duplicate": is_dup,
            "pred_duplicate": pred_dup,
            "similarity": round(sim, 4),
            "correct": pred_dup == is_dup,
        })

    true_dups = [d["true_duplicate"] for d in dup_results]
    pred_dups = [d["pred_duplicate"] for d in dup_results]
    dup_acc = float(sum(d["correct"] for d in dup_results) / len(DUP_PAIRS))
    dp, dr, df, _ = precision_recall_fscore_support(true_dups, pred_dups, average="binary", zero_division=0)
    mean_dup_sim = float(np.mean(dup_sims)) if dup_sims else 0.0
    mean_non_dup_sim = float(np.mean(non_dup_sims)) if non_dup_sims else 0.0
    sim_margin = mean_dup_sim - mean_non_dup_sim

    # 6. Confidence & Uncertainty Routing Analysis
    n_uncertain = int(rows_df["uncertain"].sum())
    uncertain_rate = float(n_uncertain / len(TEST_SET))
    confident_mask = ~rows_df["uncertain"]
    confident_exact = float(rows_df.loc[confident_mask, "correct"].mean()) if confident_mask.sum() > 0 else 0.0
    uncertain_exact = float(rows_df.loc[rows_df["uncertain"], "correct"].mean()) if n_uncertain > 0 else 0.0

    low_score_triggers = sum(1 for r in preds if r["uncertain"] and "low similarity" in r["uncertain_reason"])
    margin_triggers = sum(1 for r in preds if r["uncertain"] and "close call" in r["uncertain_reason"])

    # 7. Latency and Speed Benchmarks
    total_eval_time = float(sum(latencies_ms) / 1000.0)
    latency_stats = {
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "median_ms": round(float(np.median(latencies_ms)), 2),
        "p90_ms": round(float(np.percentile(latencies_ms, 90)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "min_ms": round(float(np.min(latencies_ms)), 2),
        "max_ms": round(float(np.max(latencies_ms)), 2),
        "total_eval_time_s": round(total_eval_time, 3),
        "throughput_rps": round(float(len(TEST_SET) / max(total_eval_time, 1e-6)), 1),
    }

    # 8. Model Runtime Profile
    model_info = clf.get_model_info() if hasattr(clf, "get_model_info") else {
        "huggingface_repo": HUGGINGFACE_MODEL_NAME,
        "device": str(clf.device) if hasattr(clf, "device") else "cpu",
    }

    # Misclassified rows
    misclassified_df = rows_df[~rows_df["correct"]].copy()

    return {
        # Backward compatible keys
        "report": report_df,
        "rows": rows_df,
        "exact_match": exact_match,
        "primary_acc": primary_acc,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "urgent_p": float(up),
        "urgent_r": float(ur),
        "urgent_f1": float(uf),
        "dup_acc": dup_acc,
        "n": len(TEST_SET),

        # Expanded Metrics
        "model_info": model_info,
        "latency": latency_stats,
        "overall": {
            "exact_set_match": exact_match,
            "primary_label_acc": primary_acc,
            "hamming_loss": hl,
            "jaccard_similarity": jaccard_samples,
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "segmented": {
            "single_n": n_single,
            "single_exact_match": single_exact,
            "single_primary_acc": single_primary,
            "multi_n": n_multi,
            "multi_exact_match": multi_exact,
            "multi_primary_acc": multi_primary,
            "multi_partial_match": multi_partial,
        },
        "urgency": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": float(up),
            "recall": float(ur),
            "specificity": float(specificity),
            "f1": float(uf),
            "balanced_accuracy": float(balanced_acc),
            "keyword_triggers": kw_count,
            "semantic_triggers": sem_count,
        },
        "duplicates": {
            "accuracy": dup_acc,
            "precision": float(dp),
            "recall": float(dr),
            "f1": float(df),
            "mean_dup_similarity": mean_dup_sim,
            "mean_non_dup_similarity": mean_non_dup_sim,
            "similarity_margin": sim_margin,
            "pairs_detail": pd.DataFrame(dup_results),
        },
        "uncertainty": {
            "flagged_count": n_uncertain,
            "flagged_rate": uncertain_rate,
            "confident_exact_match": confident_exact,
            "uncertain_exact_match": uncertain_exact,
            "low_score_triggers": low_score_triggers,
            "margin_triggers": margin_triggers,
        },
        "misclassified": misclassified_df,
    }


def print_expanded_report(res: Dict[str, Any]) -> None:
    """Print an expanded, structured terminal performance evaluation report."""
    border = "=" * 78
    divider = "-" * 78

    print("\n" + border)
    print(" 🎫 SUPPORT TICKET CLASSIFIER — EXPANDED PERFORMANCE REPORT")
    print(border)

    # Model & System Details
    m_info = res["model_info"]
    print("\n[1] MODEL & RUNTIME ARCHITECTURE")
    print(f"  • Hugging Face Repository : {m_info.get('huggingface_repo', HUGGINGFACE_MODEL_NAME)}")
    print(f"  • Direct Transformers     : {m_info.get('direct_transformers', True)} (AutoTokenizer + AutoModel)")
    print(f"  • Hidden Dimension        : {m_info.get('hidden_dimension', 384)}")
    print(f"  • Total Parameters        : {m_info.get('total_parameters', 22713216):,}")
    print(f"  • Compute Device          : {m_info.get('device', 'cpu')}")
    print(f"  • Normalization / Pooling : L2 Normalized / Attention Mask Mean-Pooling")

    # Latency & Speed
    lat = res["latency"]
    print("\n[2] INFERENCE LATENCY & THROUGHPUT BENCHMARK")
    print(f"  • Median Latency          : {lat['median_ms']} ms")
    print(f"  • Mean Latency            : {lat['mean_ms']} ms (P90: {lat['p90_ms']} ms, P95: {lat['p95_ms']} ms, P99: {lat['p99_ms']} ms)")
    print(f"  • Min / Max Latency       : {lat['min_ms']} ms / {lat['max_ms']} ms")
    print(f"  • Evaluation Throughput   : {lat['throughput_rps']} tickets / second")
    print(f"  • Total Inference Duration: {lat['total_eval_time_s']} seconds ({res['n']} benchmark tickets)")

    # Overall Metrics
    ov = res["overall"]
    print("\n[3] MULTI-LABEL CLASSIFICATION PERFORMANCE")
    print(f"  • Exact-Set Match (Subset): {ov['exact_set_match']:.2%}")
    print(f"  • Primary-Label Accuracy  : {ov['primary_label_acc']:.2%}")
    print(f"  • Hamming Loss            : {ov['hamming_loss']:.4f} (lower is better)")
    print(f"  • Jaccard Similarity (IoU): {ov['jaccard_similarity']:.2%}")
    print(f"  • Macro F1 / Micro F1     : {ov['macro_f1']:.3f} / {ov['micro_f1']:.3f}")
    print(f"  • Weighted F1             : {ov['weighted_f1']:.3f}")

    # Per-Class Breakdown Table
    print("\n[4] PER-CLASS PRECISION, RECALL & F1 METRICS")
    print(divider)
    print(res["report"].to_string())
    print(divider)

    # Single vs Multi-issue Breakdown
    seg = res["segmented"]
    print("\n[5] SINGLE-ISSUE VS MULTI-ISSUE SEGMENTED ACCURACY")
    print(f"  • Single-Issue Tickets (N={seg['single_n']}) : Exact Match = {seg['single_exact_match']:.2%}, Primary Acc = {seg['single_primary_acc']:.2%}")
    print(f"  • Multi-Issue Tickets  (N={seg['multi_n']})  : Exact Match = {seg['multi_exact_match']:.2%}, Primary Acc = {seg['multi_primary_acc']:.2%}")
    print(f"  • Multi-Issue Partial Match Rate      : {seg['multi_partial_match']:.2%}")

    # Urgency Detection Breakdown
    urg = res["urgency"]
    print("\n[6] URGENCY DETECTION & ESCALATION ANALYSIS")
    print(f"  • Confusion Matrix        : TP={urg['tp']} | FP={urg['fp']} | TN={urg['tn']} | FN={urg['fn']}")
    print(f"  • Precision / Sensitivity : {urg['precision']:.2f} / {urg['recall']:.2f} (Specificity: {urg['specificity']:.2f})")
    print(f"  • Urgency F1 Score        : {urg['f1']:.2f} (Balanced Acc: {urg['balanced_accuracy']:.2f})")
    print(f"  • Detection Trigger Source: Keyword Regex = {urg['keyword_triggers']} | Semantic Vectors = {urg['semantic_triggers']}")

    # Semantic Duplicate Detection
    dup = res["duplicates"]
    print("\n[7] SEMANTIC DUPLICATE DETECTION BENCHMARK")
    print(f"  • Pair Accuracy / F1      : {dup['accuracy']:.2%} / {dup['f1']:.2f}")
    print(f"  • Precision / Recall      : {dup['precision']:.2f} / {dup['recall']:.2f}")
    print(f"  • Mean True Duplicate Sim : {dup['mean_dup_similarity']:.3f}")
    print(f"  • Mean Non-Duplicate Sim  : {dup['mean_non_dup_similarity']:.3f}")
    print(f"  • Semantic Margin Gap     : +{dup['similarity_margin']:.3f}")

    # Uncertainty & Human-in-the-Loop Routing
    unc = res["uncertainty"]
    print("\n[8] CONFIDENCE CALIBRATION & HUMAN TRIAGE ROUTING")
    print(f"  • Tickets Flagged for Review: {unc['flagged_count']}/{res['n']} ({unc['flagged_rate']:.1%})")
    print(f"  • Accuracy on Confident Set : {unc['confident_exact_match']:.2%}")
    print(f"  • Accuracy on Uncertain Set : {unc['uncertain_exact_match']:.2%}")
    print(f"  • Flag Reason Breakdown     : Low Similarity = {unc['low_score_triggers']} | Close Margin = {unc['margin_triggers']}")

    # Misclassification Diagnostics
    wrong = res["misclassified"]
    print("\n[9] ERROR DIAGNOSTICS (MISCLASSIFICATIONS)")
    if len(wrong) > 0:
        print(divider)
        print(wrong[["text", "true", "pred", "error_type", "primary_score", "margin"]].to_string(index=False))
        print(divider)
    else:
        print("  🎉 Zero misclassifications! All test tickets matched ground truth exactly.")

    print(border + "\n")


if __name__ == "__main__":
    clf = TicketClassifier()
    eval_results = run_eval(clf)
    print_expanded_report(eval_results)
