"""FastAPI service & interactive web UI for Citation-Backed Hybrid RAG."""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.rag import CitationRAG
from app.retriever import HybridRetriever

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"

retriever = HybridRetriever(CORPUS_PATH)
rag_engine = CitationRAG(retriever=retriever)
app = FastAPI(title="RAG Citations AI", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "documents_indexed": len(retriever.chunks)}


@app.get("/sources")
def sources() -> List[Dict[str, str]]:
    return [
        {"id": c.id, "act": c.act, "section": c.section, "title": c.title}
        for c in retriever.chunks
    ]


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    res = rag_engine.ask(req.question, top_k=req.top_k)
    return {
        "question": req.question,
        "answer": res.answer,
        "citations": res.citations,
        "abstained": res.abstained,
        "confidence": res.confidence,
        "retrieved": [
            {
                "citation": h.chunk.citation,
                "title": h.chunk.title,
                "score": h.final_score,
            }
            for h in res.hits
        ],
    }


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RAG Citations AI</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; background: #0d1117; color: #e6edf3; }
    h1 { color: #a78bfa; }
    .box { display: flex; gap: 8px; margin: 16px 0; }
    input { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #30363d; background: #161b22; color: #e6edf3; font-size: 15px; }
    button { padding: 12px 20px; border-radius: 8px; border: 0; background: #7c3aed; color: white; font-weight: 600; cursor: pointer; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; margin-top: 16px; }
    .badge { display: inline-block; background: #21262d; color: #38bdf8; padding: 4px 10px; border-radius: 999px; font-size: 12px; margin: 4px 6px 0 0; }
  </style>
</head>
<body>
  <h1>RAG Citations AI</h1>
  <p>Ask about Indian consumer, RTI, IT, DPDP, or constitutional rights (English or Hinglish). Every claim cites its exact section, or declines when unsure.</p>
  <div class="box">
    <input id="q" placeholder="e.g. How many days for RTI reply? / Consumer shikayat kaise kare?" />
    <button onclick="ask()">Ask</button>
  </div>
  <div id="out"></div>
  <script>
    async function ask() {
      const q = document.getElementById('q').value;
      if (!q.trim()) return;
      const r = await fetch('/ask', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({question: q}) });
      const d = await r.json();
      const cites = d.citations.map(c => `<span class="badge">${c}</span>`).join('');
      document.getElementById('out').innerHTML = `
        <div class="card">
          <div><strong>${d.abstained ? 'Declined (Low Evidence)' : 'Cited Answer'}</strong> · confidence ${d.confidence}</div>
          <p>${d.answer}</p>
          <div>${cites}</div>
        </div>`;
    }
  </script>
</body>
</html>"""
