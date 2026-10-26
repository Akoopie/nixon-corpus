"""Query the Nixon corpus: hybrid retrieval + an LLM-written, cited answer.

Retrieval fuses keyword (FTS5) and semantic (sqlite-vec) rankings with
Reciprocal Rank Fusion, then Claude reads the top segments and writes a cited,
date-and-speaker-attributed answer suitable for decision-archaeology.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python src/query.py "What was said about the opening to China?"
    python src/query.py "..." --k 20 --no-llm     # just show retrieved segments
"""
from __future__ import annotations

import argparse
import os
import sys

from db import connect
from index import embed  # reuse the same local embedding model

RRF_K = 60  # standard RRF damping constant


def fts_rank(conn, query: str, limit: int) -> list[int]:
    # Build a safe FTS5 MATCH: keep only alphanumeric tokens and quote each one,
    # so punctuation in the question (?, commas, quotes) can't break FTS syntax.
    import re
    terms = re.findall(r"[A-Za-z0-9]+", query)
    if not terms:
        return []
    fts_query = " OR ".join(f'"{t}"' for t in terms)
    rows = conn.execute(
        """SELECT rowid FROM segments_fts
           WHERE segments_fts MATCH ?
           ORDER BY bm25(segments_fts) LIMIT ?""",
        (fts_query, limit),
    ).fetchall()
    return [r[0] for r in rows]


def vec_rank(conn, query: str, limit: int) -> list[int]:
    qvec = embed([query])[0].tobytes()
    rows = conn.execute(
        """SELECT segment_id FROM segments_vec
           WHERE embedding MATCH ? AND k = ?
           ORDER BY distance""",
        (qvec, limit),
    ).fetchall()
    return [r[0] for r in rows]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, sid in enumerate(ranking):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (k + rank + 1)
    return [sid for sid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def hydrate(conn, seg_ids: list[int]) -> list[dict]:
    if not seg_ids:
        return []
    placeholders = ",".join("?" * len(seg_ids))
    rows = conn.execute(
        f"""SELECT s.segment_id, s.conversation_id, s.speaker, s.start_ts, s.end_ts, s.text,
                   c.date, c.location, c.participants, c.title
            FROM segments s LEFT JOIN conversations c
              ON s.conversation_id = c.conversation_id
            WHERE s.segment_id IN ({placeholders})""",
        seg_ids,
    ).fetchall()
    by_id = {r["segment_id"]: dict(r) for r in rows}
    return [by_id[i] for i in seg_ids if i in by_id]


def retrieve(conn, query: str, k: int) -> list[dict]:
    pool = max(k * 3, 30)
    fused = rrf([fts_rank(conn, query, pool), vec_rank(conn, query, pool)])
    return hydrate(conn, fused[:k])


def format_context(segs: list[dict]) -> str:
    lines = []
    for i, s in enumerate(segs, 1):
        when = s.get("date") or "date?"
        loc = s.get("location") or ""
        conv = s.get("conversation_id") or "?"
        ts = f"{s.get('start_ts', 0):.0f}s"
        lines.append(
            f"[{i}] conv {conv} | {when} {loc} | {s.get('speaker','UNKNOWN')} @ {ts}\n"
            f"    {s.get('text','').strip()}"
        )
    return "\n".join(lines)


def llm_answer(query: str, segs: list[dict]) -> str:
    import anthropic

    context = format_context(segs)
    system = (
        "You are a presidential-history research assistant analyzing transcribed "
        "Nixon White House tape segments. Answer ONLY from the provided segments. "
        "Cite every claim with its [n] index and include the date and speaker. "
        "If the segments are insufficient, say so plainly. When relevant, note how "
        "a topic evolves chronologically across segments. Flag any apparent gaps or "
        "anomalies a historian might want to investigate."
    )
    user = f"Question: {query}\n\nRetrieved segments:\n{context}"

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    resp = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="natural-language question")
    ap.add_argument("--k", type=int, default=12, help="segments to retrieve (default: 12)")
    ap.add_argument("--no-llm", action="store_true", help="show retrieved segments only, skip Claude")
    args = ap.parse_args()

    conn = connect()
    segs = retrieve(conn, args.query, args.k)
    conn.close()

    if not segs:
        print("No matching segments. Is the corpus indexed? (run src/index.py)")
        return 1

    print("=== Retrieved segments ===")
    print(format_context(segs))

    if args.no_llm:
        return 0
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n(ANTHROPIC_API_KEY not set -> skipping LLM answer. Use --no-llm to silence this.)")
        return 0

    print("\n=== Answer ===")
    print(llm_answer(args.query, segs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
