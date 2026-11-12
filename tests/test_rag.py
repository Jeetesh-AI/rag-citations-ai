"""Tests for Hybrid Retriever, citation grounding, abstention, and FastAPI endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.rag import CitationRAG
from app.retriever import HybridRetriever

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"


def test_hybrid_retrieval_english_and_hinglish() -> None:
    retriever = HybridRetriever(CORPUS_PATH)
    hits_en = retriever.search("time limit for RTI reply", top_k=3)
    assert hits_en[0].chunk.id == "rti-2005-s7"

    hits_hi = retriever.search("consumer shikayat kaise kare", top_k=3)
    assert any(h.chunk.id == "cpa-2019-s35" for h in hits_hi)


def test_citation_rag_answers_with_exact_section() -> None:
    rag = CitationRAG(HybridRetriever(CORPUS_PATH))
    res = rag.ask("What is the punishment for identity theft under IT Act?")
    assert not res.abstained
    assert "Information Technology Act, 2000, Section 66C" in res.citations
    assert "Section 66C" in res.answer


def test_citation_rag_declines_out_of_scope_query() -> None:
    rag = CitationRAG(HybridRetriever(CORPUS_PATH))
    res = rag.ask("What is the recipe for butter chicken with naan?")
    assert res.abstained
    assert res.citations == []


def test_fastapi_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert len(client.get("/sources").json()) == 10

    r = client.post("/ask", json={"question": "Right to withdraw consent for personal data"})
    assert r.status_code == 200
    data = r.json()
    assert data["abstained"] is False
    assert "Digital Personal Data Protection Act, 2023, Section 6" in data["citations"]
