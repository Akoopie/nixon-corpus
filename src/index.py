"""Index transcript JSON files into the searchable corpus.

For each segment we:
  * insert the row into `segments`
  * mirror its text into the FTS5 index `segments_fts`
  * compute a Qwen3-Embedding-0.6B vector and store it in `segments_vec`

Embeddings run locally on CPU via sentence-transformers (no network, no API).

Re-indexing a conversation replaces its previous rows, so this is safe to re-run.
Transcripts whose conversation_id is not in the NARA manifest are skipped.

Usage:
    python src/index.py                              # index every JSON in the transcripts dir
    python src/index.py path/to/37-wht-conversation-692-003.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from db import connect, init_schema, EMBED_DIM
from paths import TRANSCRIPT_DIR

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model {EMBED_MODEL} (first run downloads weights)...")
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
        dim = _model.get_sentence_embedding_dimension()
        if dim != EMBED_DIM:
            print(f"WARNING: model dim {dim} != schema EMBED_DIM {EMBED_DIM}. "
                  f"Update EMBED_DIM in db.py and recreate the DB.")
    return _model


def embed(texts: list[str]):
    import numpy as np
    model = get_model()
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 64)
    return np.asarray(vecs, dtype="float32")


def delete_conversation(cur, conv_id: str) -> int:
    """Remove a conversation's rows from segments, the FTS index, and the vector index."""
    rows = cur.execute(
        "SELECT segment_id, text FROM segments WHERE conversation_id = ?", (conv_id,)
    ).fetchall()
    # External-content FTS5 tables need an explicit 'delete' carrying the old text.
    cur.executemany(
        "INSERT INTO segments_fts (segments_fts, rowid, text) VALUES ('delete', ?, ?)",
        [(r[0], r[1]) for r in rows],
    )
    cur.executemany("DELETE FROM segments_vec WHERE segment_id = ?", [(r[0],) for r in rows])
    cur.execute("DELETE FROM segments WHERE conversation_id = ?", (conv_id,))
    return len(rows)


def index_file(conn, path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    conv_id = data.get("conversation_id")
    if not conn.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conv_id,)).fetchone():
        print(f"  {path.name}: conversation_id {conv_id!r} not in NARA manifest, skipping")
        return 0
    segments = [s for s in data.get("segments", []) if (s.get("text") or "").strip()]
    if not segments:
        print(f"  {path.name}: no non-empty segments, skipping")
        return 0

    # Embed before touching the DB so a slow or failed embed never leaves the
    # conversation half-replaced.
    vecs = embed([s["text"] for s in segments])

    cur = conn.cursor()
    replaced = delete_conversation(cur, conv_id)
    inserted_ids = []
    for s in segments:
        cur.execute(
            """INSERT INTO segments (conversation_id, speaker, start_ts, end_ts, text)
               VALUES (?,?,?,?,?)""",
            (conv_id, s.get("speaker", "UNKNOWN"), s.get("start"), s.get("end"), s["text"]),
        )
        inserted_ids.append(cur.lastrowid)

    # FTS mirror
    cur.executemany(
        "INSERT INTO segments_fts (rowid, text) VALUES (?, ?)",
        [(sid, s["text"]) for sid, s in zip(inserted_ids, segments)],
    )

    # Vector index
    cur.executemany(
        "INSERT INTO segments_vec (segment_id, embedding) VALUES (?, ?)",
        [(sid, vec.tobytes()) for sid, vec in zip(inserted_ids, vecs)],
    )

    conn.commit()
    note = f" (replaced {replaced} old rows)" if replaced else ""
    print(f"  {path.name}: indexed {len(segments)} segments{note}")
    return len(segments)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="transcript JSON files (default: all in data/transcripts)")
    args = ap.parse_args()

    files = args.files or sorted(TRANSCRIPT_DIR.glob("*.json"))
    if not files:
        print("No transcript JSON files found. Run src/transcribe.py first.")
        return 1

    conn = connect()
    init_schema(conn)
    total = 0
    for path in files:
        total += index_file(conn, path)
    grand = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    print(f"Indexed {total} segments this run; corpus now holds {grand} segments.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
