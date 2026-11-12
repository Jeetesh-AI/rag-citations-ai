"""Citation-strict RAG synthesizer with explicit abstention on low evidence."""

from dataclasses import dataclass
import os
from typing import List, Optional

import httpx

from app.retriever import HybridRetriever, RetrievalHit


@dataclass
class RAGResponse:
    answer: str
    citations: List[str]
    abstained: bool
    confidence: float
    hits: List[RetrievalHit]


class CitationRAG:
    """Generates strictly cited answers or declines when evidence is insufficient."""

    def __init__(
        self,
        retriever: HybridRetriever,
        confidence_threshold: float = 0.16,
    ) -> None:
        self.retriever = retriever
        self.confidence_threshold = confidence_threshold
        self.llm_base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.llm_api_key = os.getenv("LLM_API_KEY", "").strip()
        self.llm_model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    def _synthesize_extractive(self, hits: List[RetrievalHit]) -> str:
        relevant = [h for h in hits if h.final_score >= self.confidence_threshold]
        if not relevant:
            relevant = hits[:1]
        lines = []
        for h in relevant[:2]:
            lines.append(f"{h.chunk.text} [{h.chunk.citation}]")
        return " ".join(lines)

    def _synthesize_with_llm(self, question: str, hits: List[RetrievalHit]) -> Optional[str]:
        if not self.llm_base_url or not self.llm_api_key:
            return None
        context = "\n".join(
            f"- [{h.chunk.citation}] ({h.chunk.title}): {h.chunk.text}"
            for h in hits
        )
        prompt = (
            "Answer the question using ONLY the statutory excerpts below. "
            "Cite the exact [Act, Section] after every claim. "
            "If the excerpts do not answer the question, reply exactly: ABSTAIN.\n\n"
            f"Excerpts:\n{context}\n\nQuestion: {question}"
        )
        try:
            resp = httpx.post(
                f"{self.llm_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.llm_api_key}"},
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 220,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text
        except Exception:
            return None

    def ask(self, question: str, top_k: int = 3) -> RAGResponse:
        hits = self.retriever.search(question, top_k=top_k)
        top_score = hits[0].final_score if hits else 0.0

        if not hits or top_score < self.confidence_threshold:
            return RAGResponse(
                answer=(
                    "I cannot answer this reliably because no verified statutory section "
                    "in the knowledge base matches your query with sufficient confidence."
                ),
                citations=[],
                abstained=True,
                confidence=round(top_score, 4),
                hits=hits,
            )

        relevant_hits = [h for h in hits if h.final_score >= self.confidence_threshold]
        llm_answer = self._synthesize_with_llm(question, relevant_hits)
        if llm_answer and "ABSTAIN" in llm_answer.upper():
            return RAGResponse(
                answer=(
                    "I cannot answer this reliably because the retrieved sections "
                    "do not directly address your question."
                ),
                citations=[],
                abstained=True,
                confidence=round(top_score, 4),
                hits=hits,
            )

        answer = llm_answer if llm_answer else self._synthesize_extractive(relevant_hits)
        citations = [h.chunk.citation for h in relevant_hits[:2]]
        return RAGResponse(
            answer=answer,
            citations=citations,
            abstained=False,
            confidence=round(top_score, 4),
            hits=hits,
        )
