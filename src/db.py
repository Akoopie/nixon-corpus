"""Shared database schema and connection helpers for the Nixon corpus.

The corpus lives in a single SQLite file with three logical layers:

  conversations   - one row per NARA-catalogued conversation (the manifest spine)
  segments        - one row per diarized, timestamped transcript segment
  segments_fts    - FTS5 full-text index over segment text  (keyword search)
  segments_vec    - sqlite-vec vector index over segment embeddings (semantic)

Hybrid retrieval fuses segments_fts and segments_vec with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from paths import CORPUS_DB

# Dimensionality of Qwen3-Embedding-0.6B output vectors.
EMBED_DIM = 1024

DEFAULT_DB = CORPUS_DB


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    """Open the corpus DB with sqlite-vec loaded and sane pragmas."""
    import sqlite_vec  # imported here so db.py is importable without the dep

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes if they do not yet exist."""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,   -- NARA conversation number
            title           TEXT,
            date            TEXT,               -- ISO date if parseable
            start_time      TEXT,
            end_time        TEXT,
            device          TEXT,               -- recording device
            location        TEXT,               -- Oval Office, EOB, etc.
            participants    TEXT,               -- raw participant string
            summary         TEXT                -- NARA descriptive summary
        );

        CREATE TABLE IF NOT EXISTS segments (
            segment_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT REFERENCES conversations(conversation_id),
            speaker         TEXT,               -- diarization label or resolved name
            start_ts        REAL,               -- seconds into the recording
            end_ts          REAL,
            text            TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_segments_conv ON segments(conversation_id);

        -- Keyword search. Content-less external-content FTS keyed to segments.
        CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
            text,
            content='segments',
            content_rowid='segment_id'
        );

        -- Semantic search. One vector per segment, keyed by segment_id.
        CREATE VIRTUAL TABLE IF NOT EXISTS segments_vec USING vec0(
            segment_id INTEGER PRIMARY KEY,
            embedding  FLOAT[{EMBED_DIM}]
        );
        """
    )
    conn.commit()
