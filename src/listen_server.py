"""Local listening-session server: play each clip, type what you hear, mark who said it.

Serves <root>/session (clips.json + audio/) at http://localhost:8765 and appends every answer to
<root>/session/labels.jsonl (one JSON object per clip; a later answer for the same clip wins).

    python src/listen_server.py <root>
"""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(sys.argv[1]) / "session"
PAGE = Path(__file__).with_name("listen_page.html")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.path = self.path.split("?")[0]  # cache-busting query strings
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/clips.json":
            return self._send(200, (ROOT / "clips.json").read_bytes(), "application/json")
        if self.path == "/voices.json":
            f = ROOT / "voices.json"
            return self._send(200, f.read_bytes() if f.exists() else b"{}", "application/json")
        if self.path.startswith("/voices/") and self.path.endswith(".wav"):
            f = ROOT / "voices" / Path(self.path).name
            if f.exists():
                return self._send(200, f.read_bytes(), "audio/wav")
        if self.path == "/labels.json":
            labels = {}
            f = ROOT / "labels.jsonl"
            if f.exists():
                for l in f.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        x = json.loads(l)
                        labels[x["id"]] = x
            return self._send(200, json.dumps(labels).encode(), "application/json")
        if self.path.startswith(("/audio/", "/audio_clean/")) and self.path.endswith(".wav"):
            f = ROOT / self.path.split("/")[1] / Path(self.path).name
            if f.exists():
                return self._send(200, f.read_bytes(), "audio/wav")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/label":
            return self._send(404, b"not found", "text/plain")
        x = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        x["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(ROOT / "labels.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
        self._send(200, b'{"ok":true}', "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("Listening session: open http://localhost:8765  (Ctrl+C to stop; answers save as you go)")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
