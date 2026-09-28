# tests/unit/test_evaluator.py
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from services.rag.evaluator import RAGEvaluator


def test_high_context_precision_and_faithfulness():
    """Verify grounded response with relevant context yields high precision and faithfulness."""
    evaluator = RAGEvaluator()

    query = "Who is Roboute Guilliman?"
    contexts = [
        "Roboute Guilliman is the Primarch of the Ultramarines.",
        "Guilliman serves as the Lord Commander of the Imperium.",
    ]
    response = "Roboute Guilliman is the Primarch of the Ultramarines and Lord Commander."

    scores = evaluator.evaluate(query, response, contexts)

    assert scores["context_precision"] == 1.0
    assert scores["faithfulness"] == 1.0


def test_low_faithfulness_hallucination_detection():
    """Verify ungrounded words in response drop the faithfulness score."""
    evaluator = RAGEvaluator()

    query = "Who is Vulkan?"
    contexts = ["Vulkan is the Primarch of the Salamanders."]
    # Words like "flies", "jetpack", "defeated", "necron" > 3 chars and absent from context
    response = "Vulkan flies a jetpack and defeated the Necron King on Mars."

    scores = evaluator.evaluate(query, response, contexts)

    assert scores["faithfulness"] < 0.5


def test_empty_contexts_and_query_boundary_conditions():
    """Verify evaluator handles empty contexts gracefully without raising zero-division errors."""
    evaluator = RAGEvaluator()

    # Empty retrieved contexts
    scores = evaluator.evaluate("Who is Vulkan?", "Response text.", [])
    assert scores["context_precision"] == 0.0
    assert scores["faithfulness"] == 0.0

    # Short query under word-length threshold (>3 chars)
    scores = evaluator.evaluate("Who is", "Response text.", ["Context passage."])
    assert scores["context_precision"] == 1.0