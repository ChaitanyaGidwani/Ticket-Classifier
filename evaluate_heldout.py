"""Comprehensive evaluation on independent held-out support ticket dataset.

Reports:
  1. Primary accuracy & exact-set multi-label match
  2. Per-class precision, recall, F1, and support
  3. Primary category confusion matrix
  4. Urgency detection precision, recall, F1, and false positive analysis
  5. Semantic duplicate detection accuracy on held-out paraphrase/negative pairs
  6. Inference latency benchmarks (median, p95, p99 ms)
"""
import argparse
import time
from typing import Any, Dict, List, Optional

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
from tests.heldout_test_data import (
    ALL_HELDOUT_DATA,
    HELDOUT_DUP_PAIRS,
    TEST_SET_HELDOUT,
    VALIDATION_SET,
)


def run_heldout_eval(
    clf: TicketClassifier,
    dataset: Optional[List[Dict[str, Any]]] = None,
    dup_pairs: Optional[List[tuple]] = None,
    split_name: str = "Held-Out Test Set",
) -> Dict[str, Any]:
    """Run full evaluation suite on a held-out dataset."""
    target_data = dataset if dataset is not None else TEST_SET_HELDOUT
    target_dups = dup_pairs if dup_pairs is not None else HELDOUT_DUP_PAIRS

    preds: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    for t in target_data:
        t0 = time.perf_counter()
        r = clf.classify(t["text"], check_duplicates=False)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)
        preds.append(r)

        true_set = set(t["labels"])
        pred_set = set(r["labels"])
        is_exact = true_set == pred_set
        primary_correct = r["primary"] in true_set

        error_type = "None"
        if not is_exact:
            if not primary_correct:
                error_type = "Wrong Primary"
            elif pred_set < true_set:
                error_type = "Missed Label"
            elif pred_set > true_set:
                error_type = "Extra Label"
            else:
                error_type = "Mixed"

        rows.append({
            "text": t["text"],
            "true": ", ".join(t["labels"]),
            "pred": ", ".join(r["labels"]),
            "correct": is_exact,
            "primary": r["primary"],
            "primary_correct": primary_correct,
            "error_type": error_type,
            "true_urgent": t["urgent"],
            "pred_urgent": r["urgent"],
            "urgent_correct": t["urgent"] == r["urgent"],
            "urgent_reason": r.get("urgent_reason", ""),
            "uncertain": r["uncertain"],
            "score": round(r["scores"][r["primary"]], 3),
            "margin": round(r["margin"], 3),
            "latency_ms": round(lat_ms, 2),
            "is_multi": len(t["labels"]) > 1,
        })

    rows_df = pd.DataFrame(rows)

    # Multi-label classification report
    mlb = MultiLabelBinarizer(classes=LABELS)
    yt = mlb.fit_transform([t["labels"] for t in target_data])
    yp = mlb.transform([r["labels"] for r in preds])

    report_dict = classification_report(
        yt, yp, target_names=LABELS, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).T.round(3)

    exact_match = float(sum(r["correct"] for r in rows) / len(target_data))
    primary_acc = float(sum(r["primary_correct"] for r in rows) / len(target_data))
    hl = float(hamming_loss(yt, yp))
    jaccard_samples = float(jaccard_score(yt, yp, average="samples", zero_division=0))

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(yt, yp, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(yt, yp, average="weighted", zero_division=0)

    # Primary label confusion matrix (on single-label tickets)
    single_indices = [i for i, t in enumerate(target_data) if len(t["labels"]) == 1]
    if single_indices:
        y_true_single = [target_data[i]["labels"][0] for i in single_indices]
        y_pred_single = [preds[i]["primary"] for i in single_indices]
        cm_primary = confusion_matrix(y_true_single, y_pred_single, labels=LABELS)
        cm_df = pd.DataFrame(cm_primary, index=[f"True {l}" for l in LABELS], columns=[f"Pred {l}" for l in LABELS])
    else:
        cm_df = pd.DataFrame()

    # Urgency analysis
    true_urgents = [t["urgent"] for t in target_data]
    pred_urgents = [r["urgent"] for r in preds]
    up, ur, uf, _ = precision_recall_fscore_support(true_urgents, pred_urgents, average="binary", zero_division=0)
    cm_urg = confusion_matrix(true_urgents, pred_urgents, labels=[False, True])
    tn, fp, fn, tp = int(cm_urg[0, 0]), int(cm_urg[0, 1]), int(cm_urg[1, 0]), int(cm_urg[1, 1])
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Duplicates evaluation
    dup_eval_rows = []
    for a, b, is_dup in target_dups:
        sim = clf.similarity(a, b)
        pred_dup = sim >= clf.dup_threshold
        dup_eval_rows.append({
            "ticket_a": a, "ticket_b": b,
            "true_dup": is_dup, "pred_dup": pred_dup,
            "similarity": round(sim, 3), "correct": pred_dup == is_dup,
        })
    dup_acc = float(sum(d["correct"] for d in dup_eval_rows) / len(target_dups)) if target_dups else 0.0
    dp, dr, df, _ = precision_recall_fscore_support(
        [d["true_dup"] for d in dup_eval_rows],
        [d["pred_dup"] for d in dup_eval_rows],
        average="binary", zero_division=0,
    )

    # Latencies
    latency_stats = {
        "median_ms": round(float(np.median(latencies_ms)), 2),
        "mean_ms": round(float(np.mean(latencies_ms)), 2),
        "p90_ms": round(float(np.percentile(latencies_ms, 90)), 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 2),
        "throughput_rps": round(float(len(target_data) / (sum(latencies_ms) / 1000.0)), 1),
    }

    # Uncertainty triage
    unc_count = int(rows_df["uncertain"].sum())
    unc_rate = float(unc_count / len(target_data))
    conf_mask = ~rows_df["uncertain"]
    conf_acc = float(rows_df.loc[conf_mask, "correct"].mean()) if conf_mask.sum() > 0 else 0.0

    return {
        "split_name": split_name,
        "n_samples": len(target_data),
        "report": report_df,
        "rows": rows_df,
        "confusion_matrix": cm_df,
        "exact_match": exact_match,
        "primary_acc": primary_acc,
        "hamming_loss": hl,
        "jaccard_similarity": jaccard_samples,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "urgency": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": float(up), "recall": float(ur), "specificity": float(spec), "f1": float(uf),
        },
        "duplicates": {
            "accuracy": dup_acc, "precision": float(dp), "recall": float(dr), "f1": float(df),
            "detail": pd.DataFrame(dup_eval_rows),
        },
        "latency": latency_stats,
        "uncertainty": {
            "flagged_count": unc_count, "flagged_rate": unc_rate, "confident_exact_match": conf_acc,
        },
        "misclassified": rows_df[~rows_df["correct"]].copy(),
    }


def print_heldout_report(res: Dict[str, Any]) -> None:
    border = "=" * 78
    divider = "-" * 78

    print("\n" + border)
    print(f" 🎯 HELDOUT EVALUATION REPORT — {res['split_name']} (N={res['n_samples']})")
    print(border)

    print("\n[1] ACCURACY & MULTI-LABEL PERFORMANCE")
    print(f"  • Primary-Label Accuracy  : {res['primary_acc']:.2%}")
    print(f"  • Exact-Set Match (Subset): {res['exact_match']:.2%}")
    print(f"  • Macro F1 / Micro F1     : {res['macro_f1']:.3f} / {res['micro_f1']:.3f}")
    print(f"  • Hamming Loss            : {res['hamming_loss']:.4f}")
    print(f"  • Jaccard Similarity (IoU): {res['jaccard_similarity']:.2%}")

    print("\n[2] PER-CLASS PRECISION, RECALL & F1")
    print(divider)
    print(res["report"].to_string())
    print(divider)

    if not res["confusion_matrix"].empty:
        print("\n[3] PRIMARY CATEGORY CONFUSION MATRIX (Single-Issue Tickets)")
        print(divider)
        print(res["confusion_matrix"].to_string())
        print(divider)

    urg = res["urgency"]
    print("\n[4] URGENCY DETECTION & ESCALATION ANALYSIS")
    print(f"  • Precision / Recall / F1 : {urg['precision']:.2f} / {urg['recall']:.2f} / {urg['f1']:.2f}")
    print(f"  • Specificity             : {urg['specificity']:.2f}")
    print(f"  • Confusion Matrix        : TP={urg['tp']} | FP={urg['fp']} | TN={urg['tn']} | FN={urg['fn']}")

    dup = res["duplicates"]
    print("\n[5] SEMANTIC DUPLICATE DETECTION")
    print(f"  • Accuracy / F1 Score     : {dup['accuracy']:.2%} / {dup['f1']:.2f}")
    print(f"  • Precision / Recall      : {dup['precision']:.2f} / {dup['recall']:.2f}")

    lat = res["latency"]
    print("\n[6] INFERENCE SPEED BENCHMARK (CPU)")
    print(f"  • Median Latency          : {lat['median_ms']} ms")
    print(f"  • Mean / P95 / P99 Latency: {lat['mean_ms']} ms / {lat['p95_ms']} ms / {lat['p99_ms']} ms")
    print(f"  • Throughput              : {lat['throughput_rps']} tickets/sec")

    unc = res["uncertainty"]
    print("\n[7] UNCERTAINTY & TRIAGE ROUTING")
    print(f"  • Flagged for Review Rate : {unc['flagged_count']}/{res['n_samples']} ({unc['flagged_rate']:.1%})")
    print(f"  • Confident Set Accuracy  : {unc['confident_exact_match']:.2%}")

    wrong = res["misclassified"]
    print(f"\n[8] ERROR DIAGNOSTICS ({len(wrong)} Misclassified Tickets)")
    if len(wrong) > 0:
        print(divider)
        print(wrong[["text", "true", "pred", "error_type", "score"]].head(15).to_string(index=False))
        if len(wrong) > 15:
            print(f"... and {len(wrong) - 15} more misclassified cases.")
        print(divider)

    print(border + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on held-out test data.")
    parser.add_argument("--split", choices=["test", "val", "all"], default="test",
                        help="Which held-out split to evaluate (default: test).")
    args = parser.parse_args()

    clf = TicketClassifier()
    if args.split == "val":
        data = VALIDATION_SET
        name = "Held-Out Validation Set (Tuning)"
    elif args.split == "all":
        data = ALL_HELDOUT_DATA
        name = "All Held-Out Data (Combined)"
    else:
        data = TEST_SET_HELDOUT
        name = "Held-Out Test Set (Final Reporting)"

    results = run_heldout_eval(clf, dataset=data, split_name=name)
    print_heldout_report(results)
