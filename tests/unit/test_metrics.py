# tests/unit/test_metrics.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from core.metrics import RAG_FAITHFULNESS, RAG_CONTEXT_PRECISION, CHAT_REQUESTS_TOTAL


def test_custom_prometheus_metrics_record():
    """Verify Prometheus metrics increment and record histogram samples without error."""
    # Observe sample evaluation values
    RAG_CONTEXT_PRECISION.observe(0.85)
    RAG_FAITHFULNESS.observe(0.95)
    CHAT_REQUESTS_TOTAL.labels(status="success").inc()

    # Collect and assert metric samples
    precision_samples = RAG_CONTEXT_PRECISION.collect()[0].samples
    assert any(s.name == "aegis_rag_context_precision_count" and s.value >= 1.0 for s in precision_samples)

    faithfulness_samples = RAG_FAITHFULNESS.collect()[0].samples
    assert any(s.name == "aegis_rag_faithfulness_count" and s.value >= 1.0 for s in faithfulness_samples)