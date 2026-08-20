# services/agent/tools.py
from typing import Dict, Any, List
from services.rag.retriever import RAGRetriever

_retriever = RAGRetriever()

def search_knowledge_base(query: str, limit: int = 3, score_threshold: float = 0.25) -> str:
    """
    Agent tool to query the Aegis vector database for Warhammer lore / domain context.
    Returns formatted string context for LLM consumption.
    """
    results = _retriever.search(query=query, limit=limit, score_threshold=score_threshold)
    if not results:
        return "No relevant context found in knowledge base."

    formatted_context = []
    for idx, hit in enumerate(results, 1):
        source = hit.get("metadata", {}).get("source", "unknown")
        score = hit.get("score", 0.0)
        formatted_context.append(f"[{idx}] (Score: {score:.2f} | Source: {source})\n{hit['text']}")

    return "\n\n".join(formatted_context)

# Map available tools for agent invocation
AGENT_TOOLS = {
    "search_knowledge_base": search_knowledge_base
}