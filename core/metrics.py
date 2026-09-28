# core/metrics.py
from prometheus_client import Counter, Histogram

# 1. Total Request Counter
CHAT_REQUESTS_TOTAL = Counter(
    "aegis_chat_requests_total",
    "Total number of processed chat requests",
    ["status"]
)

# 2. Context Precision Histogram
RAG_CONTEXT_PRECISION = Histogram(
    "aegis_rag_context_precision",
    "Context precision score distribution of retrieved chunks",
    buckets=[0.0, 0.25, 0.5, 0.75, 1.0]
)

# 3. Faithfulness Score Histogram
RAG_FAITHFULNESS = Histogram(
    "aegis_rag_faithfulness",
    "Faithfulness score distribution of agent responses",
    buckets=[0.0, 0.25, 0.5, 0.75, 1.0]
)

# 4. Agent Processing Latency
CHAT_LATENCY_SECONDS = Histogram(
    "aegis_chat_latency_seconds",
    "Total latency for handling chat endpoint requests in seconds"
)