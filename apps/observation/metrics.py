# app/metrics.py
from prometheus_client import Histogram, Counter

QDRANT_SEARCH_LATENCY = Histogram(
    "aegis_qdrant_search_seconds",
    "Time spent executing vector search in Qdrant",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

OLLAMA_GEN_LATENCY = Histogram(
    "aegis_ollama_llm_seconds",
    "Time spent generating responses via Ollama LLM",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0)
)

CHAT_REQUEST_COUNTER = Counter(
    "aegis_chat_requests_total",
    "Total RAG chat requests processed",
    ["status"]
)