"""Lightweight support-ticket classifier using Hugging Face transformers directly (no LLMs).

Hugging Face Model:
  Repository: sentence-transformers/all-MiniLM-L6-v2
  URL: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
  Architecture: MiniLM-L6 (6 layers, 384 hidden dimension, 22.7M parameters)
  Pooling: Mean pooling over attention mask with L2 normalization

Pipeline per ticket:
  1. Split into issue segments (multi-issue support)
  2. Embed whole message + segments via Hugging Face transformer directly
  3. Label score = mean of top-2 cosine sims to that label's seed examples
  4. Softmax -> confidence; flag uncertain via low score / small margin
  5. Urgency = keyword rules OR similarity to urgent prototypes
  6. Duplicates = cosine similarity against previously seen tickets
"""
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Explicit Hugging Face model repository identifier
HUGGINGFACE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = HUGGINGFACE_MODEL_NAME  # Backward compatibility alias

LABELS = ["Billing", "Technical Support", "Security", "Cancellation", "Feature Request"]

SEEDS = {
    "Billing": [
        "I was charged twice for my subscription this month",
        "My invoice shows the wrong amount",
        "I need a refund for last month's payment",
        "Why did the price of my plan go up?",
        "My credit card was declined when I tried to pay",
        "Please send me a copy of my receipt for tax purposes",
        "There is an unknown charge on my bill",
        "How do I update my payment method?",
        "Our payment failed and my account is suspended",
        "There are unauthorized or fraudulent charges on my card",
    ],
    "Technical Support": [
        "The app crashes every time I open it",
        "I can't log in, the page keeps loading forever",
        "The website is showing a 500 error",
        "Notifications stopped working after the latest update",
        "The file upload fails with an error message",
        "The dashboard is very slow and sometimes doesn't load",
        "I'm unable to sync my data between my phone and laptop",
        "The export button does nothing when I click it",
        "Our production dashboard or server is completely down",
    ],
    "Security": [
        "I think someone hacked my account",
        "I received a suspicious login alert from another country",
        "I want to enable two-factor authentication",
        "I got a phishing email pretending to be from your company",
        "My password was changed and I didn't do it",
        "How do you protect my personal data and is it encrypted?",
        "I want to report a security vulnerability I found",
        "Someone is using my account without my permission",
    ],
    "Cancellation": [
        "I want to cancel my subscription",
        "Please close my account and delete my data",
        "How do I stop my plan from renewing?",
        "I'd like to terminate my contract at the end of this month",
        "Cancel my membership effective immediately",
        "I no longer need the service and want to unsubscribe",
        "Please downgrade and cancel my premium plan",
        "I'm leaving your service, what is the process to cancel?",
        "Please cancel my subscription and terminate my plan",
    ],
    "Feature Request": [
        "It would be great if you added a dark mode",
        "Can you add support for exporting reports to Excel?",
        "I wish there was a way to schedule messages",
        "Please consider integrating with Slack",
        "A mobile app for Android would be really helpful",
        "It would be nice to have bulk editing for items",
        "Could you add an option to customize the dashboard?",
        "I suggest adding multi-language support",
    ],
}

URGENT_SEEDS = [
    "This is urgent, we need help immediately",
    "Our production system is completely down",
    "My account has been hacked and money is being stolen",
    "I can't access anything and have a deadline in one hour",
    "Critical issue affecting all our customers right now",
    "Fraudulent charges are appearing on my card right now",
    "Data breach, customer data may be exposed",
    "Emergency: service outage, need immediate assistance",
]

URGENT_RE = re.compile(
    r"\b(urgent|urgently|asap|immediately|right now|emergency|critical|hacked|"
    r"compromised|unauthori[sz]ed|breach|fraud\w*|stolen|outage|production|"
    r"locked out|deadline|suspended|chargeback|lawsuit|legal action|"
    r"double[- ]charged|charged twice)\b",
    re.I,
)

SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|\n+|;\s+|"
    r"\s+(?:also|additionally|and also|plus|another thing|on top of that|but first|(?:\band\s+(?:also\s+)?(?:I|we|my|our)\b)|(?:,\s*and\b))[,:]?\s+",
    re.I,
)


def split_issues(text: str, min_words: int = 3) -> List[str]:
    parts = [p.strip() for p in SPLIT_RE.split(text) if p and p.strip()]
    parts = [p for p in parts if len(p.split()) >= min_words]
    return parts or [text.strip()]


def mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token embeddings taking attention mask into account."""
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class TicketClassifier:
    def __init__(
        self,
        model_name: str = HUGGINGFACE_MODEL_NAME,
        seeds: Optional[Dict[str, List[str]]] = None,
        label_threshold: float = 0.40,   # min similarity to count a label
        uncertain_below: float = 0.40,   # best score below this -> uncertain
        margin_threshold: float = 0.05,  # top1-top2 gap below this -> uncertain
        urgent_threshold: float = 0.60,  # similarity to urgent prototypes
        dup_threshold: float = 0.70,     # cosine for duplicate tickets
        temperature: float = 0.05,       # softmax temp for confidence display
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_name = model_name
        self.hf_model_repo = model_name

        # Determine compute device
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load Hugging Face tokenizer and transformer model directly
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception:
            # Fallback to local cached files if running in restricted offline environment
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)

        self.model.to(self.device)
        self.model.eval()

        target_seeds = seeds if seeds is not None else SEEDS
        self.labels = list(target_seeds.keys())
        self.label_threshold = label_threshold
        self.uncertain_below = uncertain_below
        self.margin_threshold = margin_threshold
        self.urgent_threshold = urgent_threshold
        self.dup_threshold = dup_threshold
        self.temperature = temperature

        texts, self.seed_idx = [], {}
        for lab, examples in target_seeds.items():
            start = len(texts)
            texts.extend(examples)
            self.seed_idx[lab] = np.arange(start, len(texts))

        self.seed_emb = self._enc(texts)
        self.urgent_emb = self._enc(URGENT_SEEDS)
        self.store_emb: List[np.ndarray] = []
        self.store_text: List[str] = []

    def get_model_info(self) -> Dict[str, Any]:
        """Return explicit details on the active Hugging Face model and runtime."""
        total_params = sum(p.numel() for p in self.model.parameters())
        hidden_dim = getattr(self.model.config, "hidden_size", self.seed_emb.shape[1] if hasattr(self, "seed_emb") else 384)
        return {
            "huggingface_repo": self.model_name,
            "architecture": getattr(self.model.config, "architectures", ["BertModel"])[0] if hasattr(self.model, "config") else "Transformer",
            "hidden_dimension": hidden_dim,
            "total_parameters": total_params,
            "device": str(self.device),
            "pooling": "mean_pooling",
            "normalized": True,
            "direct_transformers": True,
        }

    # ---------- helpers ----------
    def _enc(self, texts: Union[List[str], Tuple[str, ...]], batch_size: int = 32) -> np.ndarray:
        """Encode texts directly via Hugging Face transformers with mean pooling & L2 normalization."""
        text_list = [str(t) for t in texts]
        if not text_list:
            hidden_dim = getattr(self.model.config, "hidden_size", 384)
            return np.empty((0, hidden_dim), dtype=np.float32)

        all_embs = []
        self.model.eval()
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = mean_pooling(outputs, encoded["attention_mask"])
                normed = F.normalize(pooled, p=2, dim=1)
                all_embs.append(normed.cpu().numpy().astype(np.float32))

        return np.vstack(all_embs)

    def _label_scores(self, emb: np.ndarray) -> np.ndarray:
        sims = emb @ self.seed_emb.T
        out = np.zeros((emb.shape[0], len(self.labels)))
        for j, lab in enumerate(self.labels):
            top = np.sort(sims[:, self.seed_idx[lab]], axis=1)[:, -2:]
            out[:, j] = top.mean(axis=1)
        return out

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x / self.temperature
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def reset_store(self) -> None:
        self.store_emb, self.store_text = [], []

    def similarity(self, a: str, b: str) -> float:
        e = self._enc([a, b])
        return float(e[0] @ e[1])

    # ---------- main API ----------
    def classify(self, text: str, check_duplicates: bool = True) -> Dict[str, Any]:
        text = text.strip()
        segs = split_issues(text)
        multi = len(segs) > 1
        embs = self._enc([text] + (segs if multi else []))
        whole = embs[0]
        agg = self._label_scores(embs[0:1])[0]

        issues = []
        if multi:
            seg_scores = self._label_scores(embs[1:])
            agg = np.maximum(agg, seg_scores.max(axis=0))
            for s, sc in zip(segs, seg_scores):
                j = int(sc.argmax())
                if sc[j] >= self.label_threshold:
                    issues.append({"segment": s, "label": self.labels[j], "score": float(sc[j])})

        order = np.argsort(-agg)
        primary = self.labels[int(order[0])]
        found = {primary}
        for it in issues:
            found.add(it["label"])
        labels = sorted(found, key=lambda l: -agg[self.labels.index(l)])

        conf = self._softmax(agg)
        top2 = np.sort(agg)[-2:]
        margin = float(top2[1] - top2[0])
        best = float(agg[order[0]])
        uncertain, reason = False, ""
        if best < self.uncertain_below:
            uncertain, reason = True, f"low similarity ({best:.2f})"
        elif len(labels) == 1 and margin < self.margin_threshold:
            uncertain, reason = True, f"close call between {self.labels[int(order[0])]} and {self.labels[int(order[1])]}"

        urg_sim = float((self.urgent_emb @ whole).max())
        kw = URGENT_RE.search(text)
        urgent = bool(kw) or urg_sim >= self.urgent_threshold
        urgent_reason = f"keyword '{kw.group(0)}'" if kw else (f"semantic match ({urg_sim:.2f})" if urgent else "")

        dups = []
        if check_duplicates:
            if self.store_emb:
                sims = np.stack(self.store_emb) @ whole
                for i in np.where(sims >= self.dup_threshold)[0]:
                    dups.append({"ticket": self.store_text[i], "similarity": float(sims[i])})
                dups.sort(key=lambda d: -d["similarity"])
            self.store_emb.append(whole)
            self.store_text.append(text)

        return {
            "text": text,
            "primary": primary,
            "labels": labels,
            "scores": {self.labels[j]: float(agg[j]) for j in range(len(self.labels))},
            "confidence": {self.labels[j]: float(conf[j]) for j in range(len(self.labels))},
            "margin": margin,
            "uncertain": uncertain,
            "uncertain_reason": reason,
            "issues": issues,
            "urgent": urgent,
            "urgent_reason": urgent_reason,
            "duplicates": dups,
        }

    def find_duplicate_pairs(self, texts: List[str]) -> List[Tuple[int, int, float]]:
        if len(texts) < 2:
            return []
        e = self._enc(texts)
        sims = e @ e.T
        pairs = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if sims[i, j] >= self.dup_threshold:
                    pairs.append((i, j, float(sims[i, j])))
        return sorted(pairs, key=lambda p: -p[2])
