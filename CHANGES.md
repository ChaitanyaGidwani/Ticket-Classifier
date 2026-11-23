# Summary of Changes & Architecture Evolution

This document outlines the systematic engineering changes made across Steps 0 through 4 of the Ticket-Classifier project, an honest assessment of edge cases that remain challenging, and recommended next steps when scaling to production with real customer support tickets.

---

## 1. What Changed

### Step 0: Baseline Verification & API Contract Preservation
- Benchmarked the existing system on the original 28 labeled tickets (`evaluate.py`), establishing the exact baseline metrics: 100% primary accuracy, 96.43% exact-set match, 0.987 macro F1, 0.75 urgency F1 (1 FP, 1 FN), 83.33% duplicate accuracy, and 3.28 ms median CPU latency.
- Verified and preserved all public API contracts: FastAPI endpoints (`/classify`, `/batch`, `/duplicates`, `/health`, `/thresholds`, `/evaluate`) and Streamlit tabs remain 100% backward compatible.

### Step 1: Honest Evaluation & Held-Out Test Suite
- **Corrected Dataset Documentation**: Fixed the discrepancy in the README; the original test set consists of **22 single-issue** and **6 multi-issue** tickets (not 20 + 8).
- **Constructed Held-Out Benchmark (`tests/heldout_test_data.py`)**: Authored 188 realistic customer tickets completely decoupled from the prototype seeds in `classifier.py`. The suite is split evenly into:
  - **Validation Split (94 tickets)**: Used exclusively for threshold grid search and calibration tuning.
  - **Test Split (94 tickets)**: Reserved strictly for final evaluation and reporting.
  - **Duplicate Benchmark (20 ticket pairs)**: 10 true semantic duplicate pairs and 10 hard non-duplicate foils.
- **Realistic Edge Cases**: Included short/ambiguous queries ("please help", "not working"), typos ("dshboard", "subcription"), Hinglish / conversational phrasing ("paise kat gaye", "mera password kisi ne change kar diya"), multi-issue tickets (2 distinct labels), and hard negatives ("cancel my card" $\rightarrow$ Billing, not Cancellation; "refund for hacked account" $\rightarrow$ Security + Billing).
- **Created `evaluate_heldout.py`**: Added an evaluation script reporting primary accuracy, exact-set match, per-class precision/recall/F1, confusion matrix, urgency metrics, duplicate detection accuracy, and latency profiles.

### Step 2: Accuracy Improvements & Confidence Calibration
- **Expanded Seed Prototypes**: Scaled prototype seeds from 8–9 per class to **25 diverse examples per class** in `classifier.py`, adding domain variations, conversational phrasing, and hard negative foils to disambiguate overlapping intents.
- **Automated Validation Tuning (`scripts/tune_thresholds.py`)**: Built a hyperparameter grid search optimizing macro F1 on the validation split while constraining human triage uncertainty rates to $\le 20\%$. Discovered optimal operating thresholds:
  - `label_threshold = 0.30`
  - `uncertain_below = 0.26`
  - `margin_threshold = 0.02`
  - `urgent_threshold = 0.55`
  - `dup_threshold = 0.70`
- **Calibrated Confidence Scoring**: Replaced the arbitrary, overconfident $T=0.05$ softmax with temperature-scaled softmax ($T=0.12$) coupled with empirical Platt scaling fitted to validation margin gaps. The reported `calibrated_confidence` now accurately reflects true posterior accuracy.
- **Robust Clause Segmentation (`split_issues`)**: Replaced fragile regex with multi-pattern clause segmentation that guards currency decimals (`$12.50`) and abbreviations (`e.g.`, `vs.`) while cleanly splitting on conjunction boundaries (`also`, `additionally`, `plus`, `but first`, `aur`).
- **High-Precision Urgency Engine**: Replaced naive keyword matching with a dual-signal architecture combining 12 semantic urgency prototypes (`URGENT_SEEDS`, cosine similarity $\ge 0.55$) and a contextual emergency regex. This eliminated false alarms from isolated words like "production", "deadline", or "suspended", boosting held-out urgency precision from **0.42 to 1.00** and F1 from **0.56 to 0.91** with **0 false positives**.

### Step 3: Speed, Scale & Persistence
- **Disk-Cached Embeddings**: Seed and urgency prototype vectors are serialized to `.cache/seeds_<hash>.npz`. Application cold-start and re-initialization dropped from ~850 ms to **<15 ms**.
- **Optional ONNX Runtime INT8 Quantization**: Added `scripts/export_onnx.py` to export the MiniLM model to ONNX with dynamic INT8 quantization (`.cache/minilm_int8.onnx`). When enabled (`use_onnx=True`), per-ticket CPU inference drops from 2.82 ms to **1.14 ms** (~2.5x throughput gain) and cold-start time drops by 5.5x.
- **Persistent Capped SQLite Store**: Replaced the unbounded in-memory list with a persistent SQLite database (`data/ticket_store.db`) paired with an in-memory NumPy cache. The store enforces a hard cap (10,000 entries) with LRU/FIFO eviction, ensuring ticket embeddings survive restarts without unbounded memory growth.

### Step 4: Production Hardening
- **Automated Test Suite**: Added 16 unit and integration tests across `tests/test_classifier.py` and `tests/test_api.py`, verifying output schemas, multi-issue splitting, urgency escalation, false positive rejection, persistent store recovery, input truncation, and API endpoints. All tests pass in ~3.3s.
- **Containerization & Tooling**: Added a lightweight CPU-optimized `Dockerfile`, `.dockerignore`, `pytest.ini`, pinned `requirements.txt`, and a developer `Makefile` (`make run-api`, `make run-ui`, `make test`, `make eval`, `make eval-heldout`, `make tune`, `make export-onnx`, `make docker-build`).
- **API Hardening**: Added request latency timing middleware, input validation with 20k character truncation protection, structured logging, and rate-limiting guidance for production reverse proxies.

---

## 2. What Didn't Improve (Failure Analysis & Edge Cases)

While primary accuracy rose to 93.62% and exact-set match reached 88.30% on the held-out set, analysis of the remaining 11 error diagnostics reveals specific systemic challenges:

1. **Subtle Multi-Clause Sentences Without Punctuation or Conjunctions**:
   - *Example*: `"Someone hacked into our system and now our production API is completely down!"`
   - *Target*: `Security`, `Technical Support` | *Predicted*: `Security` only.
   - *Reason*: The ticket is strongly dominated by the security breach embedding (score 0.516). Because both thoughts are fused without strong clausal punctuation, the technical support segment didn't clear the multi-label threshold (0.30) independently.
2. **Ambiguous Context-Dependent Terms**:
   - *Example*: `"We have a soft deadline next quarter for our team trial."`
   - *Target*: `Feature Request` | *Predicted*: `Cancellation` (score 0.364).
   - *Reason*: In customer support semantics, "trial" and "deadline" are frequently associated with expiring trial cancellations or payment deadlines, causing the general sentence transformer to lean toward Cancellation when the user's intent was exploratory.
3. **Implicit Intent Without Explicit Domain Keywords**:
   - *Example*: `"Can we configure custom session timeouts for sensitive financial workflows?"`
   - *Target*: `Security` | *Predicted*: `Technical Support` (score 0.336).
   - *Reason*: "Session timeouts" is parsed by the embedding model as application infrastructure/behavior rather than compliance security. Without fine-tuning on enterprise SaaS support tickets, general-purpose MiniLM weights treat session handling as a technical config.

---

## 3. Next 3 Improvements with Real User Tickets

If deploying this system to ingest real customer support streams at enterprise scale, the top three high-impact improvements would be:

### 1. Domain-Specific Contrastive Fine-Tuning (SetFit / Sentence-Transformers)
- **Problem Solved**: General-purpose `all-MiniLM-L6-v2` embeddings were trained on generic web corpora and NLI tasks, leading to semantic bleed on enterprise terms (e.g., "session timeout" vs. "security").
- **Solution**: Fine-tune the MiniLM encoder on 500–1,000 real historical support tickets using **SetFit** (Sentence Transformer Fine-Tuning) with contrastive cosine similarity loss. This aligns the embedding geometry directly with the business's support taxonomy while remaining lightweight (CPU-runnable, 22.7M parameters, zero GPU dependency).

### 2. High-Performance ANN Vector Search (FAISS / HNSW)
- **Problem Solved**: The current persistent store performs exact flat cosine similarity across up to 10,000 tickets in memory. At 100k+ ticket volumes, exact linear scans degrade response latency.
- **Solution**: Integrate `faiss-cpu` or `hnswlib` with an HNSW (Hierarchical Navigable Small World) index backed by SQLite. This maintains sub-millisecond duplicate search ($O(\log N)$) across millions of historical tickets with minimal memory overhead.

### 3. Active Learning & Human-in-the-Loop Feedback Loop
- **Problem Solved**: Static prototype seeds do not automatically adapt to newly emerging product features, billing changes, or shifts in customer phrasing.
- **Solution**: Connect the calibrated uncertainty pipeline (`uncertain=True` when score <0.26 or margin <0.02) to the customer support ticketing tool (e.g., Zendesk, Jira Service Desk). When human agents resolve or re-categorize flagged tickets, those verified examples are piped into an active learning queue to automatically enrich prototype seeds or trigger scheduled model fine-tuning.
