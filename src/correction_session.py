"""Correction session: the listener fixes round-3 Parakeet drafts of whole 3-minute windows, chunk by chunk.

    prep   <windows.jsonl> --drafts DIR --jobs jobs.jsonl --out SESSION   (cut chunks, cleaned copies, chunks.json)
    serve  SESSION                                                          (http://localhost:8766)

Each window is cut into chunks of about CHUNK_MIN-CHUNK_MAX seconds at the widest pause, with the
boundaries placed mid-pause so no audio falls between chunks; a long stretch where the model heard
nothing becomes chunks with an empty draft, since the listener may hear speech there. Each chunk
carries the draft words and their confidence so the page can highlight the ones the model doubted.
Answers append to SESSION/labels.jsonl (a later answer for the same chunk wins).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK_MIN, CHUNK_MAX = 8.0, 15.0
SR = 16000
DOUBT_SHARE = 0.15  # highlight the session's least confident 15% of draft words
CLEAN = ("highpass=f=90,equalizer=f=120:t=q:w=8:g=-30,equalizer=f=180:t=q:w=8:g=-25,equalizer=f=240:t=q:w=8:g=-30,"
         "equalizer=f=300:t=q:w=8:g=-20,equalizer=f=360:t=q:w=8:g=-20,equalizer=f=420:t=q:w=8:g=-15,lowpass=f=7000,"
         "afftdn=nr=12:nf=-35,dynaudnorm=f=150:g=15:p=0.9")
LOCATIONS = {"OVAL": "Oval Office", "EOB": "Executive Office Building", "CAB": "Cabinet Room"}


def boundaries(words: list[dict], length: float) -> list[float]:
    """Chunk edges in window seconds: walk forward, and in each CHUNK_MIN-CHUNK_MAX span cut at the
    middle of the widest pause between words (or at CHUNK_MAX inside a long silence)."""
    edges, t = [0.0], 0.0
    while length - t > CHUNK_MAX:
        gaps = [(words[k + 1]["s"] - words[k]["e"], (words[k]["e"] + words[k + 1]["s"]) / 2)
                for k in range(len(words) - 1)
                if t + CHUNK_MIN <= (words[k]["e"] + words[k + 1]["s"]) / 2 <= t + CHUNK_MAX]
        # a word that spans the whole range, or none at all: cut at the first pause after it, else at CHUNK_MAX
        if gaps:
            t = max(gaps)[1]
        else:
            nxt = [w["s"] for w in words if t + CHUNK_MAX < w["s"]]
            inside = [w for w in words if t < w["e"] <= t + CHUNK_MAX]
            t = (inside[-1]["e"] + 0.1) if inside and inside[-1]["e"] > t + CHUNK_MIN else (
                min(t + CHUNK_MAX, nxt[0] - 0.1) if nxt else t + CHUNK_MAX)
        edges.append(round(t, 2))
    edges.append(length)
    return edges


def cmd_prep(windows: Path, drafts: Path, jobs_path: Path, out: Path) -> None:
    jobs = {json.loads(l)["conv"]: json.loads(l) for l in jobs_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    for d in ("audio", "audio_clean"):
        (out / d).mkdir(parents=True, exist_ok=True)
    wins = [json.loads(l) for l in windows.read_text().splitlines() if l.strip()]
    stems = [f"{w['conv']}_{int(w['start']):05d}" for w in wins]
    drafts_by = {s: json.loads((drafts / f"{s}.json").read_text()) for s in stems}
    confs = sorted(w["c"] for d in drafts_by.values() for w in d["words"] if "c" in w)
    cut = confs[int(DOUBT_SHARE * (len(confs) - 1))] if confs else 0.0
    print(f"highlighting words with confidence below {cut:.3f}")
    session = []
    for win, stem in zip(wins, stems):
        draft = drafts_by[stem]
        words = sorted(draft["words"], key=lambda w: w["s"])
        length = win["end"] - win["start"]
        edges = boundaries(words, length)
        # decode and clean the whole window in one ffmpeg pass each, then cut chunks in Python (ffmpeg per chunk is slow)
        pcm = {}
        for d, extra in (("audio", []), ("audio_clean", ["-af", CLEAN])):
            whole = out / "_whole" / f"{stem}_{d}.wav"
            whole.parent.mkdir(exist_ok=True)
            if not whole.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(drafts / f"{stem}.flac"), *extra,
                                "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(whole)], check=True)
            with wave.open(str(whole), "rb") as r:
                pcm[d] = r.readframes(r.getnframes())
        chunks = []
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            cid = f"{stem}_{k:02d}"
            ws = [w for w in words if a <= (w["s"] + w["e"]) / 2 < b]
            for d, sig in pcm.items():
                f = out / d / f"{cid}.wav"
                if not f.exists():
                    with wave.open(str(f), "wb") as o:
                        o.setnchannels(1), o.setsampwidth(2), o.setframerate(SR)
                        o.writeframes(sig[int(a * SR) * 2:int(b * SR) * 2])
            chunks.append({"id": cid, "start": round(a, 2), "end": round(b, 2),
                           "draft": [{"w": w["w"], "unsure": w.get("c", 1.0) < cut} for w in ws]})
        j = jobs[win["conv"]]
        session.append({"window": stem, "conv": win["conv"], "band": win["band"], "tape_start": win["start"],
                        "date": j.get("date", ""), "place": LOCATIONS.get(j["location"], j["location"]),
                        "participants": [p.split(",")[0] for p in j.get("participants", [])], "chunks": chunks})
        print(f"{stem} ({win['band']}): {len(chunks)} chunks", flush=True)
    (out / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(len(w["chunks"]) for w in session)
    print(f"{len(session)} windows, {n} chunks -> {out}")


class Handler(BaseHTTPRequestHandler):
    root: Path = Path(".")
    page = Path(__file__).with_name("correction_page.html")

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send(200, self.page.read_bytes(), "text/html; charset=utf-8")
        if path == "/session.json":
            return self._send(200, (self.root / "session.json").read_bytes(), "application/json")
        if path == "/labels.json":
            labels, f = {}, self.root / "labels.jsonl"
            if f.exists():
                for l in f.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        x = json.loads(l)
                        labels[x["id"]] = x
            return self._send(200, json.dumps(labels).encode(), "application/json")
        if path.startswith(("/audio/", "/audio_clean/")) and path.endswith(".wav"):
            f = self.root / path.split("/")[1] / Path(path).name
            if f.exists():
                return self._send(200, f.read_bytes(), "audio/wav")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/label":
            return self._send(404, b"not found", "text/plain")
        x = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        x["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.root / "labels.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
        self._send(200, b'{"ok":true}', "application/json")

    def log_message(self, *a):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "serve"])
    ap.add_argument("path", type=Path, help="prep: windows.jsonl; serve: session folder")
    ap.add_argument("--drafts", type=Path)
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--page", default="correction_page.html", help="serve: page in src/ (triage_page.html for triage)")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    if a.cmd == "prep":
        cmd_prep(a.path, a.drafts, a.jobs, a.out)
        return 0
    Handler.root = a.path
    Handler.page = Path(__file__).with_name(a.page)
    print(f"Session: open http://localhost:{a.port}  (Ctrl+C to stop; answers save as you go)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
