#services/agent/tools.py
from langchain_core.tools import tool
from services.rag.retriever import RAGRetriever

_retriever = None

def get_retriever() -> RAGRetriever:
    """Create the RAG retriever lazily on first tool invocation."""
    global _retriever

    if _retriever is None:
        _retriever = RAGRetriever()

    return _retriever


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Aegis knowledge base for Warhammer 40K lore,
    Primarch information, and technical domain knowledge.
    """

    try:
        retriever = get_retriever()

        results = retriever.search(
            query=query,
            limit=3,
            score_threshold=0.25,
        )

        if not results:
            return "No relevant context found in knowledge base."

        formatted_context = []

        for idx, hit in enumerate(results, 1):
            metadata = hit.get("metadata", {})
            source = metadata.get("source", "unknown")
            score = hit.get("score", 0.0)
            text = hit.get("text", "")

            formatted_context.append(
                f"[{idx}] (Score: {score:.2f} | Source: {source})\n{text}"
            )

        return "\n\n".join(formatted_context)

    except Exception as e:
        return (
            "Knowledge base search failed. "
            f"Error: {type(e).__name__}: {e}"
        )


AGENT_TOOLS = [search_knowledge_base]