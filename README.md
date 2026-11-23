# Intelligent Customer Support Ticket Classification (No LLMs)

A lightweight AI application that automatically categorizes incoming customer support messages using a pretrained Hugging Face transformer model.

---

## 🎯 Requirements Compliance Matrix

| # | Requirement | Implementation Details | Verification / Where to Test |
|---|---|---|---|
| **1** | **Accept customer messages as text input** | Supports free-form text areas, quick-load preset examples, batch multi-line paste, CSV uploads (`text` column), and REST JSON payloads. | Streamlit Tab 1 & Tab 2, or FastAPI `POST /classify` |
| **2** | **Classify each message into 5 categories** (`Billing`, `Technical Support`, `Security`, `Cancellation`, `Feature Request`) | Few-shot nearest prototype scoring using 384-dimensional dense vectors generated directly via Hugging Face `transformers`. | `classifier.py` (`LABELS`, `SEEDS`), `app.py`, `api.py` |
| **3** | **Support messages containing multiple issues** | `split_issues()` segments compound tickets by punctuation and conjunction clauses, scoring each segment and aggregating multi-labels. | `classifier.py` (`split_issues`, `classify()["issues"]`), Streamlit Tab 1 |
| **4** | **Identify potentially urgent tickets** | Dual-channel escalation: regex pattern triggers (`URGENT_RE`) + semantic cosine similarity against `URGENT_SEEDS` prototypes (`urgent_threshold=0.60`). | `classifier.py` (`URGENT_RE`, `URGENT_SEEDS`), Streamlit 🚨 alert |
| **5** | **Display similarity scores and flag uncertain classifications** | Computes cosine similarity across all 5 classes with softmax confidence meters. Flags low top-1 score (<0.40) or narrow margin (<0.05) for human triage. | Streamlit progress bars, ⚠️ routing warning; `scores`, `confidence`, `uncertain` |
| **6** | **Detect semantically similar or duplicate tickets** | Cosine similarity against a dynamic in-memory vector store of past tickets, plus intra-batch pair discovery via `find_duplicate_pairs()`. | Streamlit Tab 1 ("Possible duplicates") & Tab 2 ("Duplicate pairs"); FastAPI `POST /duplicates` |
| **7** | **Provide a simple interface using Streamlit or FastAPI** | **Both** interfaces provided: an interactive web UI with Streamlit and a high-performance async REST API with interactive Swagger docs. | `streamlit run app.py` and `uvicorn api:app --reload` (`/docs`) |
| **8** | **Include at least 20 labeled test messages & report performance** | Built-in test suite with **28 labeled customer messages** (20 single-label, 8 multi-issue) + duplicate pair benchmarks. Reports exact match, primary accuracy, latency, confusion matrices, and per-class F1. | `python evaluate.py` or Streamlit Tab 3 |

---

## 🤖 Hugging Face Model

- **Repository**: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Architecture**: 6-layer MiniLM transformer (`BertModel`), 384 hidden dimensions, 22.7M parameters
- **Direct Transformers**: Uses `transformers.AutoTokenizer` and `transformers.AutoModel` directly with attention-masked mean pooling and L2 normalization (eliminating the heavy `sentence-transformers` wrapper).
- **Inference Speed**: ~2.7 ms per ticket on CPU (~330+ tickets/second).

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

---

### 2. Option A: Streamlit Interactive Web Application

Launch the full interactive dashboard:

```bash
streamlit run app.py
```

Features:
- **Tab 1: Single Ticket Classifier**: Test messages with quick-load presets (Billing, Tech Support, Security, Cancellation, Feature Request, Multi-Issue, Urgent, Ambiguous), confidence bars, issue segmentation, and duplicate history warnings.
- **Tab 2: Batch & Duplicate Detection**: Paste multiple messages or upload CSVs to view categorized tables and discovered duplicate clusters.
- **Tab 3: Evaluation Report**: Run the 28-ticket performance benchmark with 1 click.
- **Sidebar**: Adjust thresholds in real-time (`label_threshold`, `uncertain_below`, `margin_threshold`, `dup_threshold`).

---

### 3. Option B: FastAPI High-Performance REST API

Launch the async REST API:

```bash
uvicorn api:app --reload --port 8000
# or: python api.py
```

- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc API Reference**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Sample cURL Requests:

**Classify a Single Ticket:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month. Also, the app keeps crashing on startup."}'
```

**Classify a Batch:**
```bash
curl -X POST "http://localhost:8000/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I was charged twice this month", "My card was billed two times for this month"]}'
```

**Find Semantic Duplicate Pairs:**
```bash
curl -X POST "http://localhost:8000/duplicates" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I was charged twice", "My card got billed two times", "The app is down"], "threshold": 0.70}'
```

---

### 4. Run CLI Performance Benchmark

Execute the expanded performance evaluation suite on the 28 labeled benchmark tickets:

```bash
python evaluate.py
```

#### Benchmark Results Summary:
- **Exact-Set Multi-Label Match**: **96.43%**
- **Primary-Label Accuracy**: **100.00%**
- **Macro F1 / Micro F1**: **0.987 / 0.986**
- **Hamming Loss**: **0.0071**
- **Single-Issue Accuracy**: **100.00%** (22/22 tickets)
- **Multi-Issue Accuracy**: **83.33%** exact match, **100.00%** primary accuracy
- **Duplicate Detection Accuracy**: **83.33%** (semantic separation margin: +0.535)
- **Inference Latency**: **~2.7 ms** median per ticket on CPU
