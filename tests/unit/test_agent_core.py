# tests/unit/test_agent_core.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from unittest.mock import patch, MagicMock
from services.agent.tools import search_knowledge_base
from services.agent.core import AegisAgent


def test_search_knowledge_base_formatting():
    """Verify search_knowledge_base formats retrieved hits into structured context strings."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [
        {
            "metadata": {"source": "codex_ultramarines.pdf"},
            "score": 0.92,
            "text": "Roboute Guilliman is the Primarch of the Ultramarines.",
        }
    ]

    with patch("services.agent.tools.get_retriever", return_value=mock_retriever):
        result = search_knowledge_base.invoke({"query": "Guilliman"})

        assert "[1] (Score: 0.92 | Source: codex_ultramarines.pdf)" in result
        assert "Roboute Guilliman is the Primarch" in result


def test_search_knowledge_base_empty_results():
    """Verify tool returns fallback message when retriever returns no hits."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    with patch("services.agent.tools.get_retriever", return_value=mock_retriever):
        result = search_knowledge_base.invoke({"query": "Unknown item"})

        assert result == "No relevant context found in knowledge base."


@pytest.mark.anyio
async def test_aegis_agent_run_execution():
    """Verify AegisAgent accepts queries, processes history, and returns output dictionary."""
    agent = AegisAgent()

    # Mock the LangGraph agent executor to avoid external Ollama network calls in unit tests
    mock_response = {
        "messages": [
            MagicMock(content="Tool hit result", type="tool"),
            MagicMock(content="Roboute Guilliman leads the Ultramarines.", type="ai"),
        ]
    }

    with patch.object(agent.agent_executor, "ainvoke", return_value=mock_response):
        history = [{"role": "user", "content": "Who is the Primarch?"}]
        result = await agent.run(query="Tell me about Guilliman", history=history)

        assert "output" in result
        assert "retrieved_contexts" in result
        assert "Ultramarines" in result["output"]