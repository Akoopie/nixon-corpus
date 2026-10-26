"""Filesystem locations shared by every pipeline stage.

Code lives in the repository; bulky data (about 150 GB of audio, the corpus and
the explorer database) lives in a separate data root, by default ~/nixon-data.
Override it with the NIXON_DATA environment variable, e.g. on a cloud GPU worker.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NIXON_DATA") or Path.home() / "nixon-data")

AUDIO_DIR = DATA_DIR / "audio"              # source audio files
MANIFEST_DIR = DATA_DIR / "manifest"        # downloaded NARA CSV
TRANSCRIPT_DIR = DATA_DIR / "transcripts"   # per-conversation transcript JSON (indexed)
BENCH_DIR = DATA_DIR / "bench"              # model-comparison runs (never indexed)
VALIDATION_DIR = DATA_DIR / "validation"    # human reference transcripts for scoring ASR
CORPUS_DB = DATA_DIR / "corpus" / "nixon.db"

SECRETS_DIR = ROOT / ".secrets"             # gitignored tokens
