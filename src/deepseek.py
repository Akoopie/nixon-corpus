"""DeepSeek as the correction model (METHODS 7n): the Argon requests, unchanged, through DeepSeek's OpenAI-format API.

  run   <batch dir> [<batch dir> ...] --root <root> --model deepseek-v4-pro|deepseek-flash
        sends every request of each dir's requests.jsonl (argon.py's format) that has no answer yet, as one user
        message, and appends each answer to results.jsonl the moment it arrives, converted to Gemini's response
        format, so `argon.py fetch --local`, `correct.py apply` and the scorers read it unchanged. Resumable: a
        rerun skips answered keys and retries failed ones. Then saves per-conversation edits_argon.json (as
        `argon.py fetch --local`; argon_usage.json records the DeepSeek model); with --apply, also applies them with
        name propagation (as argon_run.py). Stops on HTTP 401/402, or before passing --max-usd (spent in these
        dirs, in-flight requests reserved at an estimate).
        --effort high|low|max|none (none = non-thinking mode, temperature 0.2); --offpeak-only starts requests
        only outside peak hours; --workers parallel requests (limits: flash 2500, pro 500)
  balance  the account balance (free; also checks the key)
  mock  a local fake DeepSeek for testing the chain without spending: replays Argon's answers for known prompts
        (--replay <batch dir with requests.jsonl + results.jsonl>), else one harmless edit per conversation;
        injects keep-alive blank lines, a 429 and a 503 (--faults), and HTTP 402 after --balance answers

Prices (published list, October 2026, USD per 1M tokens) are peak; off-peak is half. Peak is 01:00-04:00 and
06:00-10:00 UTC, Monday to Friday; a request that starts or ends in peak is costed at peak. The key is read from
$DEEPSEEK_API_KEY or .secrets/deepseek.txt and never printed.

    python src/deepseek.py run <batch dir> --root <root> --model deepseek-v4-pro [--max-usd 2] [--workers 16]
    python src/deepseek.py mock --port 8899 --replay <batch dir>     (then run ... --base-url http://127.0.0.1:8899)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import argon
from paths import SECRETS_DIR

API = "https://api.deepseek.com"
PRICES = {"deepseek-flash": {"hit": 0.006, "miss": 0.30, "out": 1.20},     # peak; off-peak is half
          "deepseek-v4-pro": {"hit": 0.044, "miss": 1.32, "out": 3.96}}
MAX_TOKENS = 131072          # answer + thinking; the API default in thinking mode is 64K
EST_OUT_PER_IN = 1.0         # reserve for an in-flight request: output (incl. thinking) per input token, generous
CHARS_PER_TOKEN = 3.5        # reservation only; DeepSeek's tokenizer, not Gemini's


def key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY") or (SECRETS_DIR / "deepseek.txt").read_text(encoding="utf-8").strip()


def is_peak(t: datetime) -> bool:
    t = t.astimezone(timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def usd(u: dict, model: str, peak: bool) -> float:
    p = PRICES[model]
    hit = u.get("prompt_cache_hit_tokens", 0) or 0
    miss = u.get("prompt_cache_miss_tokens", (u.get("prompt_tokens", 0) or 0) - hit)
    c = (hit * p["hit"] + miss * p["miss"] + (u.get("completion_tokens", 0) or 0) * p["out"]) / 1e6
    return c if peak else c / 2


def chat_body(req: dict, model: str, effort: str) -> dict:
    text = "".join(p["text"] for c in req["contents"] for p in c["parts"])
    body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": MAX_TOKENS}
    if effort == "none":
        body.update(thinking={"type": "disabled"}, temperature=0.2)   # as Argon's requests
    else:
        body.update(thinking={"type": "enabled"}, reasoning_effort=effort)  # temperature has no effect when thinking
    return body


def as_gemini(r: dict) -> dict:
    """A chat completion as a GenerateContentResponse: the answer text, finish reason, and token counts."""
    ch = (r.get("choices") or [{}])[0]
    u = r.get("usage") or {}
    think = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    return {"candidates": [{"content": {"parts": [{"text": (ch.get("message") or {}).get("content") or ""}]},
                            "finishReason": ch.get("finish_reason")}],
            "usageMetadata": {"promptTokenCount": u.get("prompt_tokens", 0),
                              "candidatesTokenCount": max((u.get("completion_tokens", 0) or 0) - think, 0),
                              "thoughtsTokenCount": think},
            "modelVersion": r.get("model")}


def load_results(bdir: Path) -> dict[str, dict]:
    """key -> last line for that key (a retried key's later line wins)."""
    f = bdir / "results.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                out[x["key"]] = x
    return out


class Runner:
    def __init__(self, model: str, effort: str, base: str, max_usd: float, offpeak_only: bool):
        import requests as rq
        self.rq, self.model, self.effort, self.url = rq, model, effort, base.rstrip("/") + "/chat/completions"
        self.max_usd, self.offpeak_only = max_usd, offpeak_only
        self.k = key()
        self.lock = threading.Lock()
        self.spent = self.reserved = 0.0
        self.stop = threading.Event()
        self.why = ""

    def reserve(self, text: str) -> float | None:
        tin = len(text) / CHARS_PER_TOKEN
        p = PRICES[self.model]
        est = (tin * p["miss"] + tin * EST_OUT_PER_IN * p["out"]) / 1e6   # at peak: an upper bound
        with self.lock:
            if self.spent + self.reserved + est > self.max_usd:
                return None
            self.reserved += est
        return est

    def halt(self, why: str) -> None:
        if not self.stop.is_set():
            self.why = why
            self.stop.set()
            print("STOP:", why, flush=True)

    def wait_offpeak(self) -> None:
        while self.offpeak_only and not self.stop.is_set():
            now = datetime.now(timezone.utc)
            if not (is_peak(now) or is_peak(now + timedelta(minutes=20))):
                return
            time.sleep(60)

    def call(self, x: dict) -> dict | None:
        """One request -> a results.jsonl line ({key, response | error, ds}), or None if not sent (stopped / budget)."""
        self.wait_offpeak()
        if self.stop.is_set():
            return None
        body = chat_body(x["request"], self.model, self.effort)
        est = self.reserve(body["messages"][0]["content"])
        if est is None:
            self.halt(f"the next request would pass the ${self.max_usd:.2f} cap (spent ${self.spent:.2f})")
            return None
        err, t0 = "", datetime.now(timezone.utc)
        try:
            for attempt in range(8):
                if self.stop.is_set():
                    return None
                t0 = datetime.now(timezone.utc)
                try:
                    r = self.rq.post(self.url, json=body, timeout=(30, 1800),
                                     headers={"Authorization": f"Bearer {self.k}", "Content-Type": "application/json"})
                except self.rq.RequestException as e:   # network drop or the server closing an unstarted request
                    err = f"{type(e).__name__}"
                    time.sleep(min(30 * (attempt + 1), 300))
                    continue
                if r.status_code == 200:
                    resp = json.loads(r.text)            # non-streaming keep-alive sends blank lines first
                    t1 = datetime.now(timezone.utc)
                    peak = is_peak(t0) or is_peak(t1)
                    cost = usd(resp.get("usage") or {}, self.model, peak)
                    with self.lock:
                        self.spent += cost
                    fin = ((resp.get("choices") or [{}])[0]).get("finish_reason")
                    line = {"key": x["key"], "response": as_gemini(resp),
                            "ds": {"model": self.model, "version": resp.get("model"), "effort": self.effort,
                                   "usage": resp.get("usage"), "finish": fin, "peak": peak, "usd": round(cost, 6),
                                   "t0": t0.isoformat(timespec="seconds"), "secs": round((t1 - t0).total_seconds(), 1)}}
                    if fin == "content_filter":
                        line["error"] = "content_filter"
                        del line["response"]
                    return line
                if r.status_code in (401, 402):
                    self.halt(f"HTTP {r.status_code} ({'bad key' if r.status_code == 401 else 'insufficient balance'})")
                    return None
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 503):
                    break                                # 400 / 422: our request is wrong; retrying will not help
                time.sleep(min(15 * 2 ** attempt, 300))
            return {"key": x["key"], "error": err or "gave up after retries", "ds": {"model": self.model, "t0": t0.isoformat(timespec="seconds")}}
        finally:
            with self.lock:
                self.reserved -= est


def cmd_run(bdirs: list[Path], root: Path, model: str, effort: str, workers: int, max_usd: float, offpeak_only: bool,
            base: str, apply: bool) -> None:
    run = Runner(model, effort, base, max_usd, offpeak_only)
    for bdir in bdirs:  # what these dirs already cost counts against the cap
        run.spent += sum((x.get("ds") or {}).get("usd", 0) for x in load_results(bdir).values())
    print(f"{model}, effort {effort}; already spent ${run.spent:.2f} in these dirs; cap ${max_usd:.2f}", flush=True)
    for bdir in bdirs:
        if run.stop.is_set():
            break
        done = load_results(bdir)
        todo = [json.loads(l) for l in (bdir / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        todo = [x for x in todo if "response" not in done.get(x["key"], {})]
        print(f"{bdir.name}: {len(todo)} requests to send ({len(done)} answered before)", flush=True)
        t_start, n_ok, n_err = time.time(), 0, 0
        with open(bdir / "results.jsonl", "a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(workers) as ex:
            for fut in as_completed([ex.submit(run.call, x) for x in todo]):   # each answer saved as soon as it arrives
                line = fut.result()
                if line is None:
                    continue
                with run.lock:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
                n_ok += "response" in line
                n_err += "error" in line
                if (n_ok + n_err) % 25 == 0:
                    print(f"  {n_ok + n_err}/{len(todo)} answered, {n_err} errors, ${run.spent:.2f} spent, "
                          f"{time.time() - t_start:.0f} s", flush=True)
        if run.stop.is_set():  # stopped part-way (balance, key, cap): the answers so far stay in results.jsonl for a rerun
            print(f"{bdir.name}: stopped part-way; nothing saved or applied; rerun to finish", flush=True)
            break
        results = {k: x.get("response") or {"error": x.get("error")} for k, x in load_results(bdir).items()}
        argon.save_results(root, results, argon.load_groups(bdir), model=f"{model} ({effort})")
        summarize(bdir)
        if apply:
            from correct import cmd_apply
            jobs = {json.loads(l)["conv"]: json.loads(l) for l in open(root / "jobs.jsonl", encoding="utf-8") if l.strip()}
            convs = [c for c in (bdir / "convs.txt").read_text(encoding="utf-8").split() if c in jobs]
            cmd_apply(root, [jobs[c] for c in convs], "edits_argon.json", True)
    print(f"spent ${run.spent:.2f}" + (f"; stopped: {run.why}" if run.why else ""), flush=True)


def summarize(bdir: Path) -> dict:
    rs = load_results(bdir)
    n_req = sum(1 for l in (bdir / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    s = {"requests": n_req, "answered": 0, "errors": {}, "finish": {}, "in": 0, "cache_hit": 0, "out": 0, "thinking": 0,
         "usd": 0.0, "peak_requests": 0, "secs_median": None}
    secs = []
    for x in rs.values():
        d = x.get("ds") or {}
        if "error" in x:
            kind = x["error"].split(":")[0]
            s["errors"][kind] = s["errors"].get(kind, 0) + 1
        else:
            s["answered"] += 1
        u = d.get("usage") or {}
        s["in"] += u.get("prompt_tokens", 0) or 0
        s["cache_hit"] += u.get("prompt_cache_hit_tokens", 0) or 0
        s["out"] += u.get("completion_tokens", 0) or 0
        s["thinking"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        s["usd"] += d.get("usd", 0) or 0
        s["peak_requests"] += bool(d.get("peak"))
        if d.get("finish"):
            s["finish"][d["finish"]] = s["finish"].get(d["finish"], 0) + 1
        if d.get("secs") is not None:
            secs.append(d["secs"])
    s["usd"] = round(s["usd"], 4)
    s["secs_median"] = sorted(secs)[len(secs) // 2] if secs else None
    (bdir / "ds_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"{bdir.name}: {s['answered']}/{n_req} answered, errors {s['errors'] or 0}, finish {s['finish']}; tokens "
          f"{s['in']:,} in ({s['cache_hit']:,} cached) / {s['out']:,} out ({s['thinking']:,} thinking); ${s['usd']:.4f}; "
          f"{s['peak_requests']} at peak; median {s['secs_median']} s", flush=True)
    return s


def cmd_balance(base: str) -> None:
    import requests as rq
    r = rq.get(base.rstrip("/") + "/user/balance", headers={"Authorization": f"Bearer {key()}"}, timeout=30)
    r.raise_for_status()
    for b in r.json().get("balance_infos", []):
        print(f"{b['currency']} {b['total_balance']} (topped up {b['topped_up_balance']}, granted {b['granted_balance']}); "
              f"available: {r.json().get('is_available')}")


# --- mock server -------------------------------------------------------------------------------------------------
def cmd_mock(port: int, replay: Path | None, faults: bool, balance: int | None) -> None:
    import re
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    def h(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    answers = {}
    if replay:
        reqs = {json.loads(l)["key"]: json.loads(l)["request"] for l in open(replay / "requests.jsonl", encoding="utf-8") if l.strip()}
        for l in open(replay / "results.jsonl", encoding="utf-8"):
            if l.strip():
                x = json.loads(l)
                cand = (x.get("response") or {}).get("candidates") or []
                text = "".join(p.get("text", "") for c in cand[:1] for p in c["content"]["parts"] if not p.get("thought"))
                answers[h("".join(p["text"] for c in reqs[x["key"]]["contents"] for p in c["parts"]))] = text
    state = {"n": 0, "ok": 0}
    lock = threading.Lock()

    def synth(text: str) -> str:
        """One harmless edit per conversation (its first block's first word, unchanged), naming the conversation."""
        edits = []
        for m in re.finditer(r"^=== (\S+) \|[^\n]*\n\[(\d+:\d+)\] (\S+)", text, re.M):
            edits.append({"conv": m.group(1), "block": m.group(2), "old": m.group(3).strip("{}[]|"),
                          "new": m.group(3).strip("{}[]|"), "confidence": "high", "reason": "sound"})
        if not edits and (m := re.search(r"^\[(\d+:\d+)\] (\S+)", text, re.M)):
            edits.append({"block": m.group(1), "old": m.group(2).strip("{}[]|"), "new": m.group(2).strip("{}[]|"),
                          "confidence": "high", "reason": "sound"})
        return "Some thoughts first, with a bracket [like this].\n" + json.dumps(edits)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code: int, obj: dict, pad: bool = False) -> None:
            data = (b"\n\n" if pad else b"") + json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.headers.get("Authorization", "").startswith("Bearer ") or not body.get("messages"):
                return self.send(400, {"error": "bad request"})
            with lock:
                state["n"] += 1
                n = state["n"]
                broke = balance is not None and state["ok"] >= balance
            if broke:
                return self.send(402, {"error": {"message": "Insufficient Balance"}})
            if faults and n == 2:
                return self.send(429, {"error": {"message": "rate limit"}})
            if faults and n == 4:
                return self.send(503, {"error": {"message": "overloaded"}})
            text = body["messages"][0]["content"]
            answer = answers.get(h(text)) or synth(text)
            with lock:
                state["ok"] += 1
            tin, tout = len(text) // 4, len(answer) // 4
            self.send(200, {"id": f"mock-{n}", "model": body["model"], "object": "chat.completion",
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": answer, "reasoning_content": "..."}}],
                            "usage": {"prompt_tokens": tin, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": tin,
                                      "completion_tokens": tout * 3, "total_tokens": tin + tout * 3,
                                      "completion_tokens_details": {"reasoning_tokens": tout * 2}}}, pad=faults)

    print(f"mock DeepSeek on http://127.0.0.1:{port} ({len(answers)} replayable answers)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "balance", "mock"])
    ap.add_argument("bdirs", type=Path, nargs="*")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--model", choices=sorted(PRICES), default="deepseek-v4-pro")
    ap.add_argument("--effort", choices=["high", "low", "max", "none"], default="high")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--offpeak-only", action="store_true")
    ap.add_argument("--base-url", default=API)
    ap.add_argument("--apply", action="store_true", help="run: apply the edits with name propagation (corpus slices)")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--faults", action="store_true")
    ap.add_argument("--balance", type=int)
    args = ap.parse_intermixed_args()   # batch dirs may follow options
    if args.cmd == "run":
        cmd_run(args.bdirs, args.root, args.model, args.effort, args.workers, args.max_usd, args.offpeak_only,
                args.base_url, args.apply)
    elif args.cmd == "balance":
        cmd_balance(args.base_url)
    else:
        cmd_mock(args.port, args.replay, args.faults, args.balance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
