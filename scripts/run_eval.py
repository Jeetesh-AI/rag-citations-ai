"""Runs the 12-case RAG evaluation suite and generates Dark/Light SVG charts."""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import CitationRAG
from app.retriever import HybridRetriever

EVAL_CASES = [
    {"q": "How many days does a CPIO have to reply to an RTI request?", "expected_id": "rti-2005-s7", "should_abstain": False},
    {"q": "Where can a consumer file a complaint for defective goods?", "expected_id": "cpa-2019-s35", "should_abstain": False},
    {"q": "What is the punishment for password or identity theft?", "expected_id": "it-2000-s66c", "should_abstain": False},
    {"q": "Can a Data Principal withdraw consent under DPDP Act?", "expected_id": "dpdp-2023-s6", "should_abstain": False},
    {"q": "Does Article 21 protect privacy and personal liberty?", "expected_id": "const-art21", "should_abstain": False},
    {"q": "What are unfair trade practices and misleading ads?", "expected_id": "cpa-2019-s2-47", "should_abstain": False},
    {"q": "Penalty for unauthorized access or virus damage to computer?", "expected_id": "it-2000-s43", "should_abstain": False},
    {"q": "Consumer shikayat aur defective goods ke liye kya kare?", "expected_id": "cpa-2019-s35", "should_abstain": False},
    {"q": "What is the capital of Mars in 2099?", "expected_id": None, "should_abstain": True},
    {"q": "How to bake a chocolate cake without eggs?", "expected_id": None, "should_abstain": True},
    {"q": "Explain NFL quarterback salary cap rules", "expected_id": None, "should_abstain": True},
    {"q": "Which Pokémon evolves into Charizard?", "expected_id": None, "should_abstain": True},
]

THEMES = {
    "light": {"bg": "#ffffff", "card": "#f8fafc", "border": "#cbd5e1", "text": "#0f172a", "muted": "#475569", "accent": "#6d28d9", "good": "#16a34a", "info": "#0284c7"},
    "dark": {"bg": "#0d1117", "card": "#161b22", "border": "#30363d", "text": "#e6edf3", "muted": "#8b949e", "accent": "#a78bfa", "good": "#4ade80", "info": "#38bdf8"},
}


def generate_eval_svg(metrics: Dict[str, float], theme: str) -> str:
    t = THEMES[theme]
    w, h = 760, 290
    bars = [
        ("Recall@3 (Answerable)", metrics["recall_at_3_pct"], t["accent"]),
        ("Citation Accuracy", metrics["citation_accuracy_pct"], t["info"]),
        ("Abstention Accuracy", metrics["abstention_accuracy_pct"], t["good"]),
    ]
    bar_svg = []
    for i, (label, val, col) in enumerate(bars):
        y = 82 + i * 62
        bw = (val / 100.0) * 420
        bar_svg.append(
            f'<text x="36" y="{y + 18}" fill="{t["text"]}" font-size="13" font-weight="600" font-family="ui-sans-serif, system-ui">{label}</text>'
            f'<rect x="240" y="{y}" width="420" height="26" rx="6" fill="{t["card"]}" stroke="{t["border"]}" />'
            f'<rect x="240" y="{y}" width="{bw:.1f}" height="26" rx="6" fill="{col}" />'
            f'<text x="675" y="{y + 18}" fill="{t["text"]}" font-size="13" font-weight="700" font-family="ui-sans-serif, system-ui">{val}%</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" rx="12" fill="{t['bg']}" stroke="{t['border']}" />
  <text x="36" y="38" fill="{t['text']}" font-size="16" font-weight="700" font-family="ui-sans-serif, system-ui">
    Hybrid RAG Evaluation Suite ({int(metrics['total_cases'])} Cases · p95 Latency {metrics['p95_ms']} ms)
  </text>
  {''.join(bar_svg)}
</svg>
"""


def generate_arch_svg(theme: str) -> str:
    t = THEMES[theme]
    w, h = 760, 220
    boxes = [
        (28, 70, 120, 75, "User Query", "EN / Hinglish", t["info"]),
        (185, 55, 165, 105, "Hybrid Search", "BM25 + TF-IDF\nRRF Reranking", t["accent"]),
        (390, 55, 160, 105, "Evidence Gate", "Threshold Check\nAbstain if Low", t["good"]),
        (590, 70, 140, 75, "Cited Answer", "[Act, Section]\nGrounded Output", t["info"]),
    ]
    nodes = []
    for x, y, bw, bh, title, sub, col in boxes:
        lines = sub.split("\n")
        sub_svg = "".join(
            f'<text x="{x + bw / 2:.1f}" y="{y + 44 + i * 16}" text-anchor="middle" fill="{t["muted"]}" font-size="11" font-family="ui-sans-serif, system-ui">{ln}</text>'
            for i, ln in enumerate(lines)
        )
        nodes.append(
            f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{t["card"]}" stroke="{col}" stroke-width="2" />'
            f'<text x="{x + bw / 2:.1f}" y="{y + 24}" text-anchor="middle" fill="{t["text"]}" font-size="13" font-weight="700" font-family="ui-sans-serif, system-ui">{title}</text>'
            f'{sub_svg}'
        )
    arrows = [(148, 107, 185, 107), (350, 107, 390, 107), (550, 107, 590, 107)]
    arr_svg = "".join(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["muted"]}" stroke-width="2" marker-end="url(#arr)" />'
        for x1, y1, x2, y2 in arrows
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="{t['muted']}" /></marker></defs>
  <rect width="{w}" height="{h}" rx="12" fill="{t['bg']}" stroke="{t['border']}" />
  <text x="28" y="32" fill="{t['text']}" font-size="15" font-weight="700" font-family="ui-sans-serif, system-ui">Citation-Strict Hybrid RAG Pipeline</text>
  {arr_svg}
  {''.join(nodes)}
</svg>
"""


def main() -> None:
    rag = CitationRAG(HybridRetriever(ROOT / "data" / "corpus.json"))
    answerable = [c for c in EVAL_CASES if not c["should_abstain"]]
    unanswerable = [c for c in EVAL_CASES if c["should_abstain"]]

    recall_hits = 0
    cite_hits = 0
    abstain_hits = 0
    latencies: List[float] = []
    rows = []

    for case in EVAL_CASES:
        t0 = time.perf_counter()
        res = rag.ask(case["q"], top_k=3)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        latencies.append(ms)

        if case["should_abstain"]:
            ok = res.abstained
            if ok:
                abstain_hits += 1
        else:
            hit_ids = [h.chunk.id for h in res.hits]
            if case["expected_id"] in hit_ids:
                recall_hits += 1
            ok = (not res.abstained) and any(
                case["expected_id"] == h.chunk.id and h.chunk.citation in res.citations
                for h in res.hits
            )
            if ok:
                cite_hits += 1

        rows.append(
            {
                "question": case["q"],
                "expected_abstain": case["should_abstain"],
                "actual_abstain": res.abstained,
                "confidence": res.confidence,
                "citations": res.citations,
                "passed": ok,
                "latency_ms": ms,
            }
        )

    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
    summary = {
        "total_cases": len(EVAL_CASES),
        "recall_at_3_pct": round((recall_hits / len(answerable)) * 100, 1),
        "citation_accuracy_pct": round((cite_hits / len(answerable)) * 100, 1),
        "abstention_accuracy_pct": round((abstain_hits / len(unanswerable)) * 100, 1),
        "p95_ms": p95,
    }

    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "img").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "eval_results.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2), encoding="utf-8"
    )

    md = [
        "# RAG Evaluation Results",
        "",
        f"- **Recall@3 (Answerable):** {summary['recall_at_3_pct']}%",
        f"- **Citation Accuracy:** {summary['citation_accuracy_pct']}%",
        f"- **Abstention Accuracy (Out-of-Scope):** {summary['abstention_accuracy_pct']}%",
        f"- **p95 Latency:** {summary['p95_ms']} ms",
        "",
        "| Question | Expected Abstain | Actual Abstain | Confidence | Passed |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md.append(
            f"| {r['question']} | {r['expected_abstain']} | {r['actual_abstain']} | {r['confidence']} | {'PASS' if r['passed'] else 'FAIL'} |"
        )
    (ROOT / "docs" / "EVALS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for theme in ("light", "dark"):
        (ROOT / "docs" / "img" / f"eval-{theme}.svg").write_text(generate_eval_svg(summary, theme), encoding="utf-8")
        (ROOT / "docs" / "img" / f"architecture-{theme}.svg").write_text(generate_arch_svg(theme), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
