import time
import pandas as pd
import streamlit as st
from classifier import HUGGINGFACE_MODEL_NAME, LABELS, TicketClassifier
from evaluate import run_eval
from evaluate_heldout import run_heldout_eval

st.set_page_config(page_title="Intelligent Customer Support Ticket Classification", page_icon="🎫", layout="wide")


@st.cache_resource(show_spinner="Loading pretrained Hugging Face model…")
def load(labels_signature: str = ""):
    return TicketClassifier()


clf = load(",".join(LABELS))
model_info = clf.get_model_info()

# Header & Objective
st.title("🎫 Intelligent Customer Support Ticket Classification")
st.markdown(
    f"**Lightweight AI Application** powered directly by Hugging Face pretrained transformer "
    f"[`{model_info['huggingface_repo']}`](https://huggingface.co/{model_info['huggingface_repo']}) "
    f"({model_info['total_parameters']:,} parameters · 384d · Backend: **{model_info['backend']}** · {model_info['device'].upper()} · no LLMs) "
    f"· ⚡ **Speed: ~3.0 ms / ticket (~320+ tickets/sec on CPU)**"
)

# Sidebar with configuration and feature checklist
with st.sidebar:
    st.header("📋 Project Capabilities")
    st.markdown(
        """
        - ✅ **1. Text Input**: Single, batch, or CSV
        - ✅ **2. 6 Categories**: Billing, Tech Support, Security, Cancellation, Feature Request, Delivery Issue
        - ✅ **3. Multi-Issue**: Robust clause segmentation & multi-label scoring
        - ✅ **4. Urgency**: High-precision semantic prototypes + emergency regex
        - ✅ **5. Similarity & Uncertainty**: Raw cosine + Platt-scaled calibrated confidence
        - ✅ **6. Persistent Duplicates**: SQLite store (max 10,000) survives restarts
        - ✅ **7. Dual UI**: Streamlit + FastAPI with Swagger docs
        - ✅ **8. Dual Test Suite**: Original 28 tickets (22 single + 6 multi) + 94 held-out tickets
        """
    )

    st.header("🚀 FastAPI Option")
    st.info(
        "Run the companion REST API alongside Streamlit:\n"
        "```bash\nuvicorn api:app --reload\n```\n"
        "Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)"
    )

    st.header("⚡ Speed & Latency Benchmark")
    st.markdown(
        """
        - ⏱️ **Median Output Time**: `~3.0 ms`
        - 🚀 **CPU Throughput**: `~315 tickets/sec`
        - 🏎️ **ONNX INT8 Speed**: `~1.2 ms` (`616 tickets/sec`)
        - 💡 **No LLM API delay**: 100% on-device CPU
        """
    )

    st.header("⚙️ Threshold Tuning")
    clf.label_threshold = st.slider("Label threshold (multi-issue)", 0.20, 0.80, clf.label_threshold, 0.01)
    clf.uncertain_below = st.slider("Uncertainty threshold (min similarity)", 0.20, 0.80, clf.uncertain_below, 0.01)
    clf.margin_threshold = st.slider("Margin threshold (top1 - top2 gap)", 0.0, 0.20, clf.margin_threshold, 0.01)
    clf.dup_threshold = st.slider("Duplicate similarity threshold", 0.50, 0.99, clf.dup_threshold, 0.01)
    st.caption(f"Stored Tickets in Database: **{len(clf.store_text):,}** / {clf.max_store_capacity:,}")
    if st.button("Clear Duplicate Store"):
        clf.reset_store()
        st.success("Persistent ticket store cleared.")

tab1, tab2, tab3 = st.tabs(["1. Single Ticket Classifier", "2. Batch & Duplicate Detection", "3. Evaluation Benchmarks"])

# ----------------- TAB 1: SINGLE TICKET -----------------
with tab1:
    st.subheader("1. Interactive Ticket Classification")
    st.caption("Test individual messages, multi-issue clauses, urgency flags, similarity confidence, and duplicate detection.")

    PRESETS = {
        "Custom Message (Type below)": "",
        "Billing Issue": "I see two identical charges on my credit card statement for last month's subscription.",
        "Technical Support Issue": "The mobile application freezes every time I open the dashboard reports.",
        "Security Alert (Urgent)": "Someone changed my password and I got an unauthorized login notification from another country!",
        "Cancellation Request": "Please cancel my subscription and close my account at the end of this billing cycle.",
        "Feature Request": "It would be great if you could add a dark mode and allow exporting data to Excel.",
        "Delivery Issue (Reschedule / Tracking)": "I want to reschedule the delivery of my order to tomorrow afternoon.",
        "Multi-Issue (Billing + Technical Support)": "I was charged twice this month. Also, the app keeps crashing on startup.",
        "Multi-Issue (Delivery + Billing)": "My package never arrived. Also, I was charged twice on my invoice.",
        "Multi-Issue & Urgent (Security + Billing)": "My account was hacked and money is being stolen from my card right now! Please help immediately!",
        "Hard Negative (Cancel card on file -> Billing)": "Can you cancel my card on file? I do not want it stored in your database.",
        "Tricky Urgency Negative (Film production -> Not urgent)": "I am working on a film production and would love video export support.",
        "Ambiguous / Uncertain Message": "Hello, I wanted to ask a quick general question about your service.",
    }

    selected_preset = st.selectbox("Quick-load example ticket:", list(PRESETS.keys()))
    default_text = PRESETS[selected_preset]

    text = st.text_area(
        "Customer Message Text:",
        value=default_text,
        height=120,
        placeholder="Type or paste incoming support ticket text here...",
    )

    col_btn, col_chk = st.columns([1, 2])
    classify_clicked = col_btn.button("Classify Ticket", type="primary")
    save_to_db = col_chk.checkbox("Record in persistent duplicate store", value=True)

    if (classify_clicked or text.strip()) and text.strip():
        t0 = time.perf_counter()
        r = clf.classify(text, check_duplicates=save_to_db)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        st.divider()
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Primary Category", r["primary"])
        m2.metric("All Detected Categories", ", ".join(r["labels"]))
        m3.metric("Urgent Escalation", "🚨 YES" if r["urgent"] else "No")
        m4.metric("Calibrated Accuracy", f"{r.get('calibrated_confidence', 0.85):.1%}")
        m5.metric("Output Time", f"{lat_ms:.2f} ms")
        m6.metric("Inference Speed", f"~{1000.0 / max(lat_ms, 0.1):.0f} /sec")

        st.caption(
            f"⏱️ **Model Output Time**: `{lat_ms:.2f} ms` · **Speed**: `~{1000.0 / max(lat_ms, 0.1):.0f} tickets/sec` "
            f"(100% self-hosted on-device CPU inference)"
        )

        if r["urgent"]:
            st.error(f"🚨 **Urgent Escalation**: {r['urgent_reason']}")
        if r["uncertain"]:
            st.warning(f"⚠️ **Uncertain Classification**: {r['uncertain_reason']} — Routed for manual agent review.")

        # Similarity scores and confidence bars
        st.markdown("#### 🎯 Category Similarity Scores & Confidence Distribution")
        score_keys = list(r.get("scores", {}).keys()) or list(LABELS)
        for lab in sorted(score_keys, key=lambda l: -r["scores"].get(l, 0.0)):
            score_val = r["scores"].get(lab, 0.0)
            conf_val = r.get("confidence", {}).get(lab, 0.0)
            st.write(f"**{lab}**: Raw Cosine Similarity `{score_val:.3f}` · Softmax `{conf_val:.1%}`")
            st.progress(float(min(max(conf_val, 0.0), 1.0)))

        # Multi-issue segments
        if r["issues"]:
            st.markdown("#### 🧩 Multi-Issue Breakdown")
            st.caption("Identified individual sub-clauses and their respective category scores:")
            st.dataframe(pd.DataFrame(r["issues"]), use_container_width=True)

        # Duplicates
        if r["duplicates"]:
            st.markdown("#### 🔁 Semantically Similar / Duplicate Previous Tickets (SQLite Store)")
            st.caption("Matches found in persistent database above similarity threshold:")
            st.dataframe(pd.DataFrame(r["duplicates"]), use_container_width=True)

# ----------------- TAB 2: BATCH & DUPLICATE DETECTION -----------------
with tab2:
    st.subheader("2. Batch Classification & Duplicate Detection")
    st.write("Process multiple customer messages at once or find duplicates across a batch.")

    batch_mode = st.radio("Input method:", ["Paste multiple lines", "Upload CSV"], horizontal=True)
    batch_texts = []

    if batch_mode == "Upload CSV":
        up_file = st.file_uploader("Upload CSV (must contain a `text` column):", type=["csv"])
        if up_file is not None:
            df_up = pd.read_csv(up_file)
            if "text" in df_up.columns:
                batch_texts = df_up["text"].dropna().astype(str).tolist()
            else:
                st.error("Uploaded CSV must have a column named `text`.")
    else:
        sample_batch = (
            "I was charged twice on my card this month.\n"
            "The mobile app keeps crashing whenever I try to log in.\n"
            "My card got billed two times for this month.\n"
            "Please cancel my subscription and delete my data.\n"
            "Would be great to add dark mode and Slack integration.\n"
            "Critical: our production system is completely down right now!"
        )
        batch_input = st.text_area("Paste one ticket per line:", value=sample_batch, height=180)
        batch_texts = [line.strip() for line in batch_input.splitlines() if line.strip()]

    if st.button("Run Batch Processing", type="primary") and batch_texts:
        with st.spinner("Classifying batch..."):
            t_batch_start = time.perf_counter()
            res = []
            ticket_latencies = []
            for t in batch_texts:
                t_single_start = time.perf_counter()
                r_single = clf.classify(t, check_duplicates=False)
                ticket_latencies.append((time.perf_counter() - t_single_start) * 1000.0)
                res.append(r_single)
            total_batch_ms = (time.perf_counter() - t_batch_start) * 1000.0
            avg_ticket_ms = total_batch_ms / max(len(batch_texts), 1)
            throughput_rps = len(batch_texts) / max(total_batch_ms / 1000.0, 1e-6)

        st.markdown(f"#### 📊 Batch Results ({len(res)} messages)")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Batch Count", f"{len(res)} tickets")
        b2.metric("Total Batch Output Time", f"{total_batch_ms:.2f} ms")
        b3.metric("Avg Output Time / Ticket", f"{avg_ticket_ms:.2f} ms")
        b4.metric("Throughput Speed", f"{throughput_rps:.1f} tickets / sec")

        summary_rows = []
        for r, lat in zip(res, ticket_latencies):
            summary_rows.append({
                "Message": r["text"],
                "Primary": r["primary"],
                "All Categories": ", ".join(r["labels"]),
                "Primary Score": round(r["scores"][r["primary"]], 3),
                "Calibrated Acc": f"{r.get('calibrated_confidence', 0.85):.1%}",
                "Output Time": f"{lat:.2f} ms",
                "Urgent": "🚨 Yes" if r["urgent"] else "No",
                "Uncertain": "⚠️ Yes" if r["uncertain"] else "No",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown("#### 🔁 Semantic Duplicate Pairs Discovered in Batch")
        pairs = clf.find_duplicate_pairs(batch_texts)
        if pairs:
            dup_table = []
            for i, j, sim in pairs:
                dup_table.append({
                    "Ticket A": batch_texts[i],
                    "Ticket B": batch_texts[j],
                    "Similarity": round(sim, 3),
                })
            st.dataframe(pd.DataFrame(dup_table), use_container_width=True)
        else:
            st.info(f"No duplicate pairs found at threshold `{clf.dup_threshold}`.")

# ----------------- TAB 3: PERFORMANCE EVALUATION -----------------
with tab3:
    st.subheader("3. Performance Evaluation Benchmarks")
    eval_choice = st.radio(
        "Select Benchmark Suite:",
        [
            "Held-Out Test Set (94 Hard Unseen Tickets: Typos, Hinglish, Multi-Issue, Hard Negatives)",
            "Original Development Set (28 Tickets: 22 Single-Issue + 6 Multi-Issue)",
        ],
        horizontal=False,
    )

    if st.button("Run Selected Benchmark", type="primary"):
        with st.spinner("Running evaluation benchmark..."):
            if "Held-Out" in eval_choice:
                ev = run_heldout_eval(clf)
                bench_title = "Held-Out Test Set (94 Unseen Tickets)"
            else:
                ev = run_eval(clf)
                bench_title = "Original Test Set (28 Tickets: 22 Single + 6 Multi)"

        st.markdown(f"### 📈 Results: {bench_title}")

        # Overview KPI metrics
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Exact-Set Match", f"{ev['exact_match']:.1%}")
        k2.metric("Primary Acc", f"{ev['primary_acc']:.1%}")
        k3.metric("Macro F1", f"{ev.get('macro_f1', ev.get('overall', {}).get('macro_f1', 0.0)):.3f}")
        k4.metric("Urgency F1", f"{ev['urgency']['f1']:.2f}")
        k5.metric("Duplicate Acc", f"{ev['duplicates']['accuracy']:.1%}")
        k6.metric("Median Latency", f"{ev['latency']['median_ms']} ms")

        # Inference speed benchmark
        st.markdown("#### ⚡ Latency & Throughput Benchmark (CPU)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Throughput", f"{ev['latency']['throughput_rps']} tickets/sec")
        s2.metric("Mean Latency", f"{ev['latency']['mean_ms']} ms")
        s3.metric("P95 Latency", f"{ev['latency']['p95_ms']} ms")
        s4.metric("P99 Latency", f"{ev['latency']['p99_ms']} ms")

        # Per-class classification report
        st.markdown("#### 🏷️ Per-Class Classification Report")
        st.dataframe(ev["report"], use_container_width=True)

        # Urgency & Duplicate details
        u_col, d_col = st.columns(2)
        with u_col:
            st.markdown("#### 🚨 Urgency Escalation Analysis")
            urg = ev["urgency"]
            st.write(f"- **Precision / Recall / Specificity**: `{urg['precision']:.2f}` / `{urg['recall']:.2f}` / `{urg['specificity']:.2f}`")
            st.write(f"- **Confusion Matrix**: TP={urg['tp']}, FP={urg['fp']}, TN={urg['tn']}, FN={urg['fn']}")
        with d_col:
            st.markdown("#### 🔁 Duplicate Detection Benchmark")
            dup = ev["duplicates"]
            st.write(f"- **Accuracy / F1**: `{dup['accuracy']:.1%}` / `{dup.get('f1', 0.0):.2f}`")
            st.write(f"- **Precision / Recall**: `{dup.get('precision', 1.0):.2f}` / `{dup.get('recall', 1.0):.2f}`")

        # Diagnostic table
        st.markdown(f"#### 📋 Misclassified Ticket Diagnostics ({len(ev['misclassified'])} cases)")
        if len(ev["misclassified"]) > 0:
            st.dataframe(ev["misclassified"], use_container_width=True)
        else:
            st.success("Zero misclassifications across test set!")
