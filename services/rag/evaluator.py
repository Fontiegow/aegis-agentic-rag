import string
from typing import List, Dict, Any

class RAGEvaluator:
    def _clean_text(self, text: str) -> str:
        """Removes punctuation and normalizes casing."""
        return text.translate(str.maketrans("", "", string.punctuation)).lower()

    def evaluate_context_precision(self, query: str, retrieved_contexts: List[str]) -> float:
        if not retrieved_contexts:
            return 0.0
            
        clean_query = self._clean_text(query)
        query_terms = [t for t in clean_query.split() if len(t) > 3]
        
        if not query_terms:
            return 1.0

        relevant_chunks = 0
        for ctx in retrieved_contexts:
            clean_ctx = self._clean_text(ctx)
            if any(term in clean_ctx for term in query_terms):
                relevant_chunks += 1
                
        return round(relevant_chunks / len(retrieved_contexts), 2)

    def evaluate_faithfulness(self, response: str, retrieved_contexts: List[str]) -> float:
        if not response or not retrieved_contexts:
            return 0.0
            
        combined_context = self._clean_text(" ".join(retrieved_contexts))
        clean_response = self._clean_text(response)
        words = [w for w in clean_response.split() if len(w) > 3]
        
        if not words:
            return 1.0
        
        matches = sum(1 for w in words if w in combined_context)
        score = matches / len(words)
        return round(score, 2)

    def evaluate(self, query: str, response: str, retrieved_contexts: List[str]) -> Dict[str, float]:
        return {
            "context_precision": self.evaluate_context_precision(query, retrieved_contexts),
            "faithfulness": self.evaluate_faithfulness(response, retrieved_contexts),
        }