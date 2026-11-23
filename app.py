import pandas as pd
import streamlit as st
from classifier import HUGGINGFACE_MODEL_NAME, LABELS, TicketClassifier
from evaluate import run_eval

st.set_page_config(page_title="Intelligent Customer Support Ticket Classification", page_icon="🎫", layout="wide")


@st.cache_resource(show_spinner="Loading pretrained Hugging Face model…")
def load():
    return TicketClassifier()


clf = load()
model_info = clf.get_model_info()

# Header & Objective
st.title("🎫 Intelligent Customer Support Ticket Classification")
st.markdown(
    f"**Lightweight AI Application** powered directly by Hugging Face pretrained transformer "
    f"[`{model_info['huggingface_repo']}`](https://huggingface.co/{model_info['huggingface_repo']}) "
    f"({model_info['total_parameters']:,} parameters · 384d · {model_info['device'].upper()} · no LLMs)"
)

# Sidebar with configuration and feature checklist
with st.sidebar:
    st.header("📋 Project Requirements")
    st.markdown(
        """
        - ✅ **1. Text Input**: Single, batch, or CSV
        - ✅ **2. 5 Categories**: Billing, Tech Support, Security, Cancellation, Feature Request
        - ✅ **3. Multi-Issue**: Segmentation & multi-label scoring
        - ✅ **4. Urgency**: Keywords + semantic prototypes
        - ✅ **5. Similarity & Uncertainty**: Scores + review flagging
        - ✅ **6. Semantic Duplicates**: Embedding cosine store
        - ✅ **7. Dual UI**: Streamlit + FastAPI
        - ✅ **8. Test Suite**: 28 labeled test messages (≥20)
        """
    )

    st.header("🚀 FastAPI Option")
    st.info(
        "Run the companion REST API alongside Streamlit:\n"
        "```bash\nuvicorn api:app --reload\n```\n"
        "Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)"
    )

    st.header("⚙️ Threshold Tuning")
    clf.label_threshold = st.slider("Label threshold (multi-issue)", 0.20, 0.80, clf.label_threshold, 0.01)
    clf.uncertain_below = st.slider("Uncertainty threshold (min similarity)", 0.20, 0.80, clf.uncertain_below, 0.01)
    clf.margin_threshold = st.slider("Margin threshold (top1 - top2 gap)", 0.0, 0.20, clf.margin_threshold, 0.01)
    clf.dup_threshold = st.slider("Duplicate similarity threshold", 0.50, 0.99, clf.dup_threshold, 0.01)
    if st.button("Clear Duplicate History"):
        clf.reset_store()
        st.success("Ticket history cleared.")

tab1, tab2, tab3 = st.tabs(["1. Single Ticket Classifier", "2. Batch & Duplicate Detection", "3. Evaluation Report (28 Test Tickets)"])

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
        "Multi-Issue (Billing + Technical Support)": "I was charged twice this month. Also, the app keeps crashing on startup.",
        "Multi-Issue & Urgent (Security + Billing)": "My account was hacked and money is being stolen from my card right now! Please help immediately!",
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

    col_btn, _ = st.columns([1, 4])
    classify_clicked = col_btn.button("Classify Ticket", type="primary")

    if (classify_clicked or text.strip()) and text.strip():
        r = clf.classify(text, check_duplicates=True)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Primary Category", r["primary"])
        m2.metric("All Detected Categories", ", ".join(r["labels"]))
        m3.metric("Urgent Escalation", "🚨 YES" if r["urgent"] else "No")
        m4.metric("Human Review Needed", "⚠️ Flagged" if r["uncertain"] else "Confident")

        if r["urgent"]:
            st.error(f"🚨 **Urgent Escalation**: {r['urgent_reason']}")
        if r["uncertain"]:
            st.warning(f"⚠️ **Uncertain Classification**: {r['uncertain_reason']} — Routed for manual agent review.")

        # Similarity scores and confidence bars
        st.markdown("#### 🎯 Category Similarity Scores & Calibrated Confidence")
        for lab in sorted(LABELS, key=lambda l: -r["scores"][l]):
            score_val = r["scores"][lab]
            conf_val = r["confidence"][lab]
            st.write(f"**{lab}**: Cosine Similarity `{score_val:.3f}` · Calibrated Confidence `{conf_val:.1%}`")
            st.progress(float(min(max(conf_val, 0.0), 1.0)))

        # Multi-issue segments
        if r["issues"]:
            st.markdown("#### 🧩 Multi-Issue Breakdown")
            st.caption("Identified individual sub-clauses and their respective category scores:")
            st.dataframe(pd.DataFrame(r["issues"]), use_container_width=True)

        # Duplicates
        if r["duplicates"]:
            st.markdown("#### 🔁 Semantically Similar / Duplicate Previous Tickets")
            st.caption("Previously seen tickets matching above the duplicate similarity threshold:")
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
            res = [clf.classify(t, check_duplicates=False) for t in batch_texts]

        st.markdown(f"#### 📊 Batch Results ({len(res)} messages)")
        summary_rows = []
        for r in res:
            summary_rows.append({
                "Message": r["text"],
                "Primary": r["primary"],
                "All Categories": ", ".join(r["labels"]),
                "Primary Score": round(r["scores"][r["primary"]], 3),
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
    st.subheader("3. Comprehensive Performance Evaluation (28 Test Tickets)")
    st.markdown(
        "Rigorous benchmark on the built-in labeled test set (**28 customer messages**, exceeding the required 20). "
        "Evaluates exact-set multi-label match, primary category accuracy, latency benchmarks, urgency escalation, "
        "and duplicate detection."
    )

    if st.button("Run Performance Benchmark", type="primary"):
        with st.spinner("Running evaluation benchmark..."):
            ev = run_eval(clf)

        # Overview KPI metrics
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Exact-Set Match", f"{ev['exact_match']:.1%}")
        k2.metric("Primary Acc", f"{ev['primary_acc']:.1%}")
        k3.metric("Macro F1", f"{ev['overall']['macro_f1']:.3f}")
        k4.metric("Urgency F1", f"{ev['urgent_f1']:.2f}")
        k5.metric("Duplicate Acc", f"{ev['dup_acc']:.1%}")
        k6.metric("Median Latency", f"{ev['latency']['median_ms']} ms")

        # Inference speed benchmark
        st.markdown("#### ⚡ Latency & Throughput Benchmark (CPU)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Throughput", f"{ev['latency']['throughput_rps']} tickets/sec")
        s2.metric("Mean Latency", f"{ev['latency']['mean_ms']} ms")
        s3.metric("P95 Latency", f"{ev['latency']['p95_ms']} ms")
        s4.metric("Total Benchmark Time", f"{ev['latency']['total_eval_time_s']} s")

        # Single vs multi issue
        st.markdown("#### 🔍 Single-Issue vs Multi-Issue Segmented Accuracy")
        seg = ev["segmented"]
        c_s, c_m = st.columns(2)
        c_s.info(
            f"**Single-Issue Tickets (N={seg['single_n']})**\n\n"
            f"- Exact Match: **{seg['single_exact_match']:.1%}**\n"
            f"- Primary Accuracy: **{seg['single_primary_acc']:.1%}**"
        )
        c_m.info(
            f"**Multi-Issue Tickets (N={seg['multi_n']})**\n\n"
            f"- Exact Match: **{seg['multi_exact_match']:.1%}**\n"
            f"- Primary Accuracy: **{seg['multi_primary_acc']:.1%}**\n"
            f"- Partial Match (≥1 label): **{seg['multi_partial_match']:.1%}**"
        )

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
            st.write(f"- **Detection Triggers**: Keyword Regex = {urg['keyword_triggers']}, Semantic Vector = {urg['semantic_triggers']}")
        with d_col:
            st.markdown("#### 🔁 Duplicate Detection Benchmark")
            dup = ev["duplicates"]
            st.write(f"- **Pair Accuracy / F1**: `{dup['accuracy']:.1%}` / `{dup['f1']:.2f}`")
            st.write(f"- **Mean Duplicate Similarity**: `{dup['mean_dup_similarity']:.3f}`")
            st.write(f"- **Mean Non-Duplicate Similarity**: `{dup['mean_non_dup_similarity']:.3f}`")
            st.write(f"- **Semantic Separation Margin**: `+{dup['similarity_margin']:.3f}`")

        # Uncertainty Triage
        st.markdown("#### 🛡️ Confidence Calibration & Review Triage")
        unc = ev["uncertainty"]
        st.write(
            f"- **Flagged for Human Review**: `{unc['flagged_count']}/{ev['n']}` ({unc['flagged_rate']:.1%})\n"
            f"- **Accuracy on Confident Tickets**: `{unc['confident_exact_match']:.1%}`\n"
            f"- **Accuracy on Uncertain Tickets**: `{unc['uncertain_exact_match']:.1%}`\n"
            f"- **Trigger Reasons**: Low Similarity = {unc['low_score_triggers']}, Close Margin = {unc['margin_triggers']}"
        )

        # Diagnostic table
        st.markdown("#### 📋 Full Per-Message Evaluation Log (28 Tickets)")
        st.dataframe(ev["rows"], use_container_width=True)
