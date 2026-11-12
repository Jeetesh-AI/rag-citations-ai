"""Hybrid retriever combining BM25, TF-IDF cosine similarity, and RRF reranking."""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Dict, List

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to", "for",
    "of", "and", "or", "by", "with", "as", "be", "can", "do", "does", "what",
    "how", "when", "who", "which", "that", "this", "any", "shall", "may", "under",
    "kya", "hai", "kaise", "kaun", "kab", "ke", "ki", "ko", "mein", "se", "aur",
}

# Lightweight Hinglish/Hindi -> English legal term normalization for bilingual queries
TERM_ALIASES = {
    "shikayat": "complaint",
    "adhikar": "rights",
    "sujana": "information",
    "soochana": "information",
    "jankari": "information",
    "samay": "days",
    "din": "days",
    "chori": "theft",
    "data": "personal data",
    "sammatti": "consent",
    "sahmati": "consent",
    "azadi": "freedom",
    "nijta": "privacy",
    "jeene": "life",
    "nakli": "defective",
    "kharab": "defective",
}


def tokenize(text: str) -> List[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    expanded: List[str] = []
    for tok in raw:
        if tok in STOPWORDS or len(tok) <= 1:
            continue
        expanded.append(tok)
        if tok in TERM_ALIASES:
            expanded.extend(TERM_ALIASES[tok].split())
    return expanded


@dataclass
class DocumentChunk:
    id: str
    act: str
    section: str
    title: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.act}, {self.section}"


@dataclass
class RetrievalHit:
    chunk: DocumentChunk
    bm25_score: float
    cosine_score: float
    rrf_score: float
    final_score: float


class HybridRetriever:
    """BM25 + TF-IDF cosine similarity fused with Reciprocal Rank Fusion."""

    def __init__(self, corpus_path: Path) -> None:
        raw_items = json.loads(corpus_path.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**item) for item in raw_items]
        self._doc_tokens: List[List[str]] = [
            tokenize(f"{c.act} {c.section} {c.title} {c.text}") for c in self.chunks
        ]
        self._doc_tf: List[Counter] = [Counter(toks) for toks in self._doc_tokens]
        self._avg_len = (
            sum(len(t) for t in self._doc_tokens) / max(1, len(self._doc_tokens))
        )
        self._df: Dict[str, int] = {}
        for tf in self._doc_tf:
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
        self._n_docs = len(self.chunks)
        self._tfidf_norms: List[float] = [
            math.sqrt(sum(self._tfidf_weight(term, count) ** 2 for term, count in tf.items()))
            for tf in self._doc_tf
        ]

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def _tfidf_weight(self, term: str, tf: int) -> float:
        idf = math.log((self._n_docs + 1) / (self._df.get(term, 0) + 1)) + 1.0
        return (1.0 + math.log(tf)) * idf if tf > 0 else 0.0

    def _bm25_scores(self, q_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
        scores: List[float] = []
        for idx, tf in enumerate(self._doc_tf):
            doc_len = len(self._doc_tokens[idx])
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf(term)
                denom = freq + k1 * (1.0 - b + b * (doc_len / max(1.0, self._avg_len)))
                score += idf * ((freq * (k1 + 1.0)) / denom)
            scores.append(score)
        return scores

    def _cosine_scores(self, q_tokens: List[str]) -> List[float]:
        q_tf = Counter(q_tokens)
        q_weights = {t: self._tfidf_weight(t, c) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0.0:
            return [0.0] * self._n_docs

        scores: List[float] = []
        for idx, tf in enumerate(self._doc_tf):
            d_norm = self._tfidf_norms[idx]
            if d_norm == 0.0:
                scores.append(0.0)
                continue
            dot = sum(w * self._tfidf_weight(t, tf.get(t, 0)) for t, w in q_weights.items())
            scores.append(dot / (q_norm * d_norm))
        return scores

    def search(self, query: str, top_k: int = 3) -> List[RetrievalHit]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        bm25 = self._bm25_scores(q_tokens)
        cosine = self._cosine_scores(q_tokens)

        bm25_rank = {
            idx: rank for rank, idx in enumerate(sorted(range(self._n_docs), key=lambda i: bm25[i], reverse=True), start=1)
        }
        cos_rank = {
            idx: rank for rank, idx in enumerate(sorted(range(self._n_docs), key=lambda i: cosine[i], reverse=True), start=1)
        }

        q_set = set(q_tokens)
        hits: List[RetrievalHit] = []
        for idx, chunk in enumerate(self.chunks):
            rrf = (1.0 / (60 + bm25_rank[idx])) + (1.0 / (60 + cos_rank[idx]))
            # Rerank boost for lexical overlap and non-zero semantic/BM25 match
            doc_set = set(self._doc_tokens[idx])
            overlap = len(q_set & doc_set) / max(1, len(q_set))
            final_score = (0.55 * cosine[idx]) + (0.25 * min(1.0, bm25[idx] / 6.0)) + (0.20 * overlap)
            if cosine[idx] == 0.0 and bm25[idx] == 0.0:
                final_score = 0.0
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    bm25_score=round(bm25[idx], 4),
                    cosine_score=round(cosine[idx], 4),
                    rrf_score=round(rrf, 5),
                    final_score=round(final_score, 4),
                )
            )

        hits.sort(key=lambda h: (h.final_score, h.rrf_score), reverse=True)
        return hits[:top_k]
