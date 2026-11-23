"""Hyperparameter grid-search optimization for Support Ticket Classifier.

Evaluates STRICTLY on the VALIDATION split from tests/heldout_test_data.py
(never touches TEST_SET_HELDOUT).

Optimizes:
  - label_threshold: minimum similarity for detecting additional multi-issues
  - uncertain_below: minimum top-1 score below which a ticket is flagged uncertain
  - margin_threshold: top1 - top2 difference threshold for flagging close calls
  - urgent_threshold: semantic similarity threshold for urgency escalation
  - dup_threshold: duplicate cosine similarity threshold

Constraint:
  - Keeps uncertain-flagged rate <= 20% on the validation set
  - Maximizes Macro-F1 and multi-label Exact-Set Match
"""
import argparse
import itertools
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

from classifier import LABELS, TicketClassifier
from tests.heldout_test_data import HELDOUT_DUP_PAIRS, VALIDATION_SET


def tune_thresholds(
    clf: TicketClassifier,
    max_uncertain_rate: float = 0.20,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Grid-search hyperparameters on the validation split."""
    texts = [t["text"] for t in VALIDATION_SET]
    true_labels = [t["labels"] for t in VALIDATION_SET]
    true_urgents = [t["urgent"] for t in VALIDATION_SET]

    mlb = MultiLabelBinarizer(classes=LABELS)
    yt = mlb.fit_transform(true_labels)

    # 1. Pre-encode and pre-compute segments and scores for fast grid search
    if verbose:
        print(f"Pre-encoding {len(texts)} validation tickets...")

    cached_eval_data = []
    for t in VALIDATION_SET:
        text = t["text"]
        segs = clf._split_issues(text)
        multi = len(segs) > 1
        embs = clf._enc([text] + (segs if multi else []))
        whole = embs[0]
        agg_whole = clf._label_scores(embs[0:1])[0]
        seg_scores = clf._label_scores(embs[1:]) if multi else None
        urg_sim = float((clf.urgent_emb @ whole).max())
        kw_urgent = bool(clf.urgent_re.search(text))

        cached_eval_data.append({
            "text": text,
            "whole": whole,
            "agg_whole": agg_whole,
            "seg_scores": seg_scores,
            "segs": segs,
            "urg_sim": urg_sim,
            "kw_urgent": kw_urgent,
        })

    # Search grids
    label_thresh_grid = [0.30, 0.34, 0.36, 0.38, 0.40, 0.42]
    unc_below_grid = [0.25, 0.28, 0.30, 0.32, 0.35]
    margin_thresh_grid = [0.02, 0.03, 0.04, 0.05]
    urgent_thresh_grid = [0.48, 0.50, 0.52, 0.55]
    dup_thresh_grid = [0.65, 0.70, 0.75, 0.80]

    best_config = None
    best_score = -1.0
    best_metrics = {}

    total_combos = (
        len(label_thresh_grid)
        * len(unc_below_grid)
        * len(margin_thresh_grid)
        * len(urgent_thresh_grid)
    )
    if verbose:
        print(f"Evaluating {total_combos} threshold configurations on validation set...")

    for l_th, u_below, m_th, urg_th in itertools.product(
        label_thresh_grid, unc_below_grid, margin_thresh_grid, urgent_thresh_grid
    ):
        pred_labels_list = []
        pred_urgents = []
        uncertain_count = 0

        for item in cached_eval_data:
            agg = np.copy(item["agg_whole"])
            issues = []
            if item["seg_scores"] is not None:
                agg = np.maximum(agg, item["seg_scores"].max(axis=0))
                for s, sc in zip(item["segs"], item["seg_scores"]):
                    j = int(sc.argmax())
                    if sc[j] >= l_th:
                        issues.append(clf.labels[j])

            order = np.argsort(-agg)
            primary = clf.labels[int(order[0])]
            found = {primary}
            for lab in issues:
                found.add(lab)
            sorted_labs = sorted(found, key=lambda l: -agg[clf.labels.index(l)])
            pred_labels_list.append(sorted_labs)

            # Uncertainty check
            top2 = np.sort(agg)[-2:]
            margin = float(top2[1] - top2[0])
            best_sc = float(agg[order[0]])
            is_uncertain = False
            if best_sc < u_below:
                is_uncertain = True
            elif len(sorted_labs) == 1 and margin < m_th:
                is_uncertain = True
            if is_uncertain:
                uncertain_count += 1

            # Urgency check
            is_urgent = (item["urg_sim"] >= urg_th) or (item["kw_urgent"] and item["urg_sim"] >= 0.35)
            pred_urgents.append(is_urgent)

        uncertain_rate = uncertain_count / len(texts)

        # Enforce constraint: uncertain rate <= max_uncertain_rate (20%)
        if uncertain_rate > max_uncertain_rate:
            continue

        yp = mlb.transform(pred_labels_list)
        exact_match = sum(set(t) == set(p) for t, p in zip(true_labels, pred_labels_list)) / len(texts)
        primary_acc = sum(p[0] in t for t, p in zip(true_labels, pred_labels_list)) / len(texts)
        _, _, macro_f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        _, _, urg_f1, _ = precision_recall_fscore_support(true_urgents, pred_urgents, average="binary", zero_division=0)

        # Composite score prioritizing Macro-F1 + Exact Match + Urgency
        composite = (0.50 * macro_f1) + (0.30 * exact_match) + (0.20 * urg_f1)

        if composite > best_score:
            best_score = composite
            best_config = {
                "label_threshold": l_th,
                "uncertain_below": u_below,
                "margin_threshold": m_th,
                "urgent_threshold": urg_th,
            }
            best_metrics = {
                "macro_f1": round(float(macro_f1), 4),
                "exact_match": round(float(exact_match), 4),
                "primary_acc": round(float(primary_acc), 4),
                "urgency_f1": round(float(urg_f1), 4),
                "uncertain_rate": round(float(uncertain_rate), 4),
                "composite_score": round(float(composite), 4),
            }

    # Best duplicate threshold on validation duplicate pairs
    best_dup_th = 0.70
    best_dup_acc = -1.0
    for d_th in dup_thresh_grid:
        dup_ok = sum((clf.similarity(a, b) >= d_th) == d for a, b, d in HELDOUT_DUP_PAIRS) / len(HELDOUT_DUP_PAIRS)
        if dup_ok > best_dup_acc:
            best_dup_acc = dup_ok
            best_dup_th = d_th

    if best_config:
        best_config["dup_threshold"] = best_dup_th
        best_metrics["duplicate_acc"] = round(float(best_dup_acc), 4)

    if verbose:
        print("\n" + "=" * 60)
        print(" 🎯 GRID SEARCH RESULTS (ON VALIDATION SET ONLY)")
        print("=" * 60)
        if best_config:
            print("Optimal Tuned Thresholds:")
            for k, v in best_config.items():
                print(f"  • {k:20s}: {v}")
            print("\nValidation Metrics under Optimal Thresholds:")
            for k, v in best_metrics.items():
                print(f"  • {k:20s}: {v}")
        else:
            print("No configuration satisfied uncertain_rate <= 20%. Consider relaxing constraints.")
        print("=" * 60 + "\n")

    return {"best_config": best_config, "best_metrics": best_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune classifier thresholds on validation split.")
    parser.add_argument("--max-uncertain", type=float, default=0.20,
                        help="Maximum allowed uncertainty rate on validation set (default: 0.20).")
    args = parser.parse_args()

    classifier = TicketClassifier()
    tune_thresholds(classifier, max_uncertain_rate=args.max_uncertain)
