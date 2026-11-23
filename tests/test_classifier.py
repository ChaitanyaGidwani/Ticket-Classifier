"""Unit and integration tests for TicketClassifier."""
import os
import shutil
import tempfile
import pytest
import numpy as np

from classifier import LABELS, TicketClassifier, split_issues


@pytest.fixture
def temp_classifier():
    """Create a temporary classifier with isolated cache and sqlite db."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_store.db")
    cache_dir = os.path.join(tmp_dir, ".cache")
    clf = TicketClassifier(cache_dir=cache_dir, db_path=db_path)
    yield clf
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_classifier_output_schema(temp_classifier):
    """Verify classifier output contains all required fields and correct types."""
    res = temp_classifier.classify("I was charged twice this month for my subscription.")
    expected_keys = [
        "text", "primary", "labels", "scores", "confidence", "calibrated_confidence",
        "margin", "uncertain", "uncertain_reason", "issues", "urgent", "urgent_reason", "duplicates"
    ]
    for key in expected_keys:
        assert key in res, f"Missing key: {key}"

    assert res["primary"] in LABELS
    assert isinstance(res["labels"], list) and len(res["labels"]) >= 1
    assert isinstance(res["scores"], dict)
    assert len(res["scores"]) == len(LABELS)
    assert isinstance(res["confidence"], dict)
    assert isinstance(res["urgent"], bool)
    assert isinstance(res["uncertain"], bool)


def test_multi_issue_decomposition(temp_classifier):
    """Verify that multi-issue tickets are decomposed and assigned multiple labels."""
    compound_text = "I was charged twice this month. Also, the app keeps crashing on startup."
    res = temp_classifier.classify(compound_text, check_duplicates=False)
    assert "Billing" in res["labels"], f"Expected Billing in {res['labels']}"
    assert "Technical Support" in res["labels"], f"Expected Technical Support in {res['labels']}"
    assert len(res["issues"]) >= 1


def test_split_issues_protection():
    """Verify clause splitter handles decimals, abbreviations, and conjunctions."""
    text1 = "Invoice #1042 was $19.99 instead of $9.99. Also, the export to CSV failed."
    parts1 = split_issues(text1)
    assert len(parts1) == 2
    assert "19.99" in parts1[0]
    assert "CSV failed" in parts1[1]

    text2 = "Please review e.g. ticket #551; we want to terminate our contract."
    parts2 = split_issues(text2)
    assert any("terminate" in p for p in parts2)


def test_urgency_escalation_true_positives(temp_classifier):
    """Verify genuine critical emergencies trigger urgency escalation."""
    critical_text = "CRITICAL OUTAGE: Our production server is down and money is being stolen!"
    res = temp_classifier.classify(critical_text, check_duplicates=False)
    assert res["urgent"] is True
    assert "outage" in res["urgent_reason"].lower() or "semantic" in res["urgent_reason"].lower()


def test_urgency_escalation_false_positive_rejection(temp_classifier):
    """Verify casual messages containing 'production', 'deadline', 'suspended' are NOT falsely flagged urgent."""
    harmless_cases = [
        "I am working on a film production and would love video export support.",
        "We have a soft deadline next quarter for our team trial.",
        "I suspended my own account temporarily while on vacation.",
        "Can I schedule messages before a deadline?",
    ]
    for text in harmless_cases:
        res = temp_classifier.classify(text, check_duplicates=False)
        assert res["urgent"] is False, f"False urgency positive on: '{text}' (reason: {res['urgent_reason']})"


def test_duplicate_detection(temp_classifier):
    """Verify duplicate detection against stored tickets."""
    t1 = "I was charged twice on my credit card this month."
    t2 = "My card got billed two times for this month's subscription."
    t3 = "How do I turn on dark mode theme?"

    # Classify t1 to add it to store
    temp_classifier.classify(t1, check_duplicates=True)

    # Classify t2 (duplicate)
    r2 = temp_classifier.classify(t2, check_duplicates=True)
    assert len(r2["duplicates"]) >= 1
    assert r2["duplicates"][0]["similarity"] >= temp_classifier.dup_threshold

    # Classify t3 (unrelated)
    r3 = temp_classifier.classify(t3, check_duplicates=True)
    assert len(r3["duplicates"]) == 0


def test_empty_and_very_long_input(temp_classifier):
    """Verify handling of short and very long input (truncation to 512 tokens)."""
    # Very short input
    short_res = temp_classifier.classify("help")
    assert short_res["primary"] in LABELS

    # Very long input (> 1,000 words)
    long_text = "The server is returning an error. " + ("word " * 1500)
    long_res = temp_classifier.classify(long_text, check_duplicates=False)
    assert long_res["primary"] in LABELS
    assert isinstance(long_res["scores"], dict)


def test_persistent_store_across_restarts():
    """Verify that stored tickets persist in SQLite and are recognized by fresh instances."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "persistent.db")
    cache_dir = os.path.join(tmp_dir, ".cache")

    try:
        # Instance 1: write ticket
        clf1 = TicketClassifier(cache_dir=cache_dir, db_path=db_path)
        clf1.classify("I was charged twice on my invoice", check_duplicates=True)
        assert len(clf1.store_text) == 1

        # Instance 2: reload from same db
        clf2 = TicketClassifier(cache_dir=cache_dir, db_path=db_path)
        assert len(clf2.store_text) == 1
        assert clf2.store_text[0] == "I was charged twice on my invoice"

        # Check duplicate recognized
        r = clf2.classify("My card was billed two times on my invoice", check_duplicates=True)
        assert len(r["duplicates"]) >= 1

        # Reset store
        clf2.reset_store()
        assert len(clf2.store_text) == 0

        # Instance 3: verify reset persisted
        clf3 = TicketClassifier(cache_dir=cache_dir, db_path=db_path)
        assert len(clf3.store_text) == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
