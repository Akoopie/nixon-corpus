"""Run the corpus Argon pass in budgeted slices through the Batch API, unattended and resumable (METHODS 7l).

  plan    hybrid requests for the given conversations (argon.py --by-day, --solo-above 2000), cut into slices of
          whole days in date order, each at most SLICE_TOKENS input tokens (larger batches were refused with HTTP
          429) -> <run>/slices/NN/{requests.jsonl, groups.json, convs.txt, estimate.json}
  run     for each slice in order, skipping fetched ones:
            budget  spent so far + this slice's estimate x the observed actual/estimate ratio must stay within
                    --budget, else stop
            submit  on HTTP 429 (spend-rate limit) wait and retry, 10 min growing to 60 min; on 402 stop
            wait    poll every 2 min until the batch is done
            fetch   per-conversation edits into the corpus; actual cost from the responses' token counts
            apply   correct.py apply with name propagation
          state in <run>/state.json, log in <run>/run.log; safe to stop and restart
  status  slices done, spent, projected total

    python src/argon_run.py plan <run dir> --root ~/nixon-data/corpus_v5 --convs convs.txt
    python src/argon_run.py run <run dir> --root ~/nixon-data/corpus_v5 --budget USD [--spent USD]
    python src/argon_run.py status <run dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import argon

SLICE_TOKENS = 3_200_000
FIRST_RATIO = 1.10      # actual/estimate assumed before any slice has been measured (corpus01a: 1.03)


def log(run: Path, msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M}  {msg}"
    print(line, flush=True)
    with open(run / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state(run: Path) -> dict:
    f = run / "state.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"spent": 0.0, "slices": {}}


def save_state(run: Path, st: dict) -> None:
    (run / "state.json").write_text(json.dumps(st, indent=1), encoding="utf-8")


def cost_of(usage: dict) -> float:
    return (usage.get("promptTokenCount", 0) / 1e6 * argon.PRICE_IN +
            ((usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0)) / 1e6 * argon.PRICE_OUT)


def cmd_plan(run: Path, root: Path, convs: list[str]) -> None:
    allr = run / "all"
    argon.cmd_requests_by_day(root, convs, allr, solo_above=2000)
    groups = json.loads((allr / "groups.json").read_text(encoding="utf-8"))
    by_day: dict[str, list[str]] = {}
    for line in open(allr / "requests.jsonl", encoding="utf-8"):
        if line.strip():
            by_day.setdefault(json.loads(line)["key"].split("#")[0], []).append(line)

    def est(lines):
        tin = tout = 0.0
        for l in lines:
            text = json.loads(l)["request"]["contents"][0]["parts"][0]["text"]
            tr = text[text.find("<transcripts>"):text.find("</transcripts>")]
            tin += len(text) / argon.CHARS_PER_TOKEN
            tout += len(tr) / argon.CHARS_PER_TOKEN * argon.OUT_PER_TRANSCRIPT
        return tin, tin / 1e6 * argon.PRICE_IN + tout / 1e6 * argon.PRICE_OUT

    slices, cur, cur_tok = [], [], 0.0
    for day in sorted(by_day):
        tok, _ = est(by_day[day])
        if cur and cur_tok + tok > SLICE_TOKENS:
            slices.append(cur); cur, cur_tok = [], 0.0
        cur.append(day); cur_tok += tok
    if cur:
        slices.append(cur)
    total = 0.0
    for k, days in enumerate(slices, 1):
        d = run / "slices" / f"{k:02d}"
        d.mkdir(parents=True, exist_ok=True)
        lines = [l for day in days for l in by_day[day]]
        keys = [json.loads(l)["key"] for l in lines]
        with open(d / "requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(lines)
        (d / "groups.json").write_text(json.dumps({k2: groups[k2] for k2 in keys}, indent=0), encoding="utf-8")
        (d / "convs.txt").write_text("\n".join(dict.fromkeys(c for k2 in keys for c in groups[k2]["convs"])) + "\n",
                                     encoding="utf-8", newline="\n")
        tok, usd = est(lines)
        total += usd
        (d / "estimate.json").write_text(json.dumps({"days": [days[0], days[-1]], "n_days": len(days), "requests": len(lines),
                                                     "input_tokens": round(tok), "usd": round(usd, 2)}), encoding="utf-8")
    print(f"{len(slices)} slices, {sum(map(len, slices))} days, estimated ${total:.2f} "
          f"(x{FIRST_RATIO} until measured: ${total * FIRST_RATIO:.2f})")


def submit_with_retry(run: Path, d: Path) -> str:
    """'ok', or 'payment' (HTTP 402: stop), after waiting out HTTP 429 spend-rate refusals."""
    import requests as rq
    wait = 600
    while True:
        try:
            argon.cmd_submit(d)
            return "ok"
        except (rq.ConnectionError, rq.Timeout) as e:  # the local network dropped: not a refusal
            log(run, f"slice {d.name}: network error ({type(e).__name__}); retrying in 5 min")
            time.sleep(300)
            continue
        except rq.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 402:
                return "payment"
            if code != 429:
                raise
            log(run, f"slice {d.name}: HTTP 429 (spend-rate limit); retrying in {wait // 60} min")
            time.sleep(wait)
            wait = min(wait + 600, 3600)


def cmd_run(run: Path, root: Path, budget: float, spent0: float) -> None:
    from correct import cmd_apply
    st = load_state(run)
    if spent0 and not st["slices"]:
        st["spent"] = spent0
    jobs = {json.loads(l)["conv"]: json.loads(l) for l in open(root / "jobs.jsonl", encoding="utf-8") if l.strip()}
    for d in sorted((run / "slices").iterdir()):
        s = st["slices"].setdefault(d.name, {"status": "planned"})
        if s["status"] == "fetched":
            continue
        est = json.loads((d / "estimate.json").read_text(encoding="utf-8"))
        if s["status"] == "planned":
            measured = [x for x in st["slices"].values() if x.get("status") == "fetched"]
            ratio = (sum(x["actual"] for x in measured) / sum(x["est"] for x in measured)) if measured else FIRST_RATIO
            projected = st["spent"] + est["usd"] * ratio
            if projected > budget:
                log(run, f"STOP before slice {d.name} ({est['days'][0]}..{est['days'][1]}): spent ${st['spent']:.2f}, "
                         f"this slice ~${est['usd'] * ratio:.2f} would pass the ${budget:.2f} budget")
                break
            res = submit_with_retry(run, d)
            if res == "payment":
                log(run, f"STOP: HTTP 402 Payment Required at slice {d.name}; the account needs credit")
                break
            s.update(status="submitted", est=est["usd"], batch=json.loads((d / "batch.json").read_text(encoding="utf-8"))["job"]["name"])
            save_state(run, st)
            log(run, f"slice {d.name} submitted: {est['days'][0]}..{est['days'][1]}, {est['requests']} requests, est ${est['usd']:.2f}")
        import requests as rq
        while True:  # submitted: wait for the batch
            try:
                state = argon.job_state(argon.get_job(d))
            except (rq.ConnectionError, rq.Timeout):
                time.sleep(120)
                continue
            if "SUCCEEDED" in state:
                break
            if any(x in state for x in ("FAILED", "CANCELLED", "EXPIRED")):
                log(run, f"STOP: slice {d.name} ended {state}")
                return
            time.sleep(120)
        argon.download_results(d)
        failed, total = argon.failed_share(d)
        if failed > total / 2:  # e.g. a slice accepted whose requests then all came back RESOURCE_EXHAUSTED
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            (d / "results.jsonl").replace(d / f"results_failed_{stamp}.jsonl")
            (d / "batch.json").replace(d / f"batch_failed_{stamp}.json")
            s.clear()
            s.update(status="planned", note=f"{failed}/{total} requests failed at {stamp}; not applied")
            save_state(run, st)
            log(run, f"STOP: slice {d.name} came back with {failed} of {total} requests failed (quota?); nothing applied")
            return
        argon.cmd_fetch(d, root, local=True)
        actual =sum(cost_of((json.loads(l).get("response") or {}).get("usageMetadata") or {})
                     for l in open(d / "results.jsonl", encoding="utf-8") if l.strip())
        convs = [c for c in (d / "convs.txt").read_text(encoding="utf-8").split() if c in jobs]
        cmd_apply(root, [jobs[c] for c in convs], "edits_argon.json", True)
        s.update(status="fetched", actual=round(actual, 4))
        st["spent"] = round(st["spent"] + actual, 4)
        save_state(run, st)
        log(run, f"slice {d.name} done: ${actual:.2f} (est ${s['est']:.2f}); {len(convs)} conversations; spent ${st['spent']:.2f} of ${budget:.2f}")
    cmd_status(run)


def cmd_status(run: Path) -> None:
    st = load_state(run)
    ests = {d.name: json.loads((d / "estimate.json").read_text(encoding="utf-8")) for d in sorted((run / "slices").iterdir())}
    done = [k for k, v in st["slices"].items() if v.get("status") == "fetched"]
    measured = [st["slices"][k] for k in done]
    ratio = (sum(x["actual"] for x in measured) / sum(x["est"] for x in measured)) if measured else FIRST_RATIO
    left = sum(e["usd"] for k, e in ests.items() if k not in done) * ratio
    print(f"{len(done)} of {len(ests)} slices done; spent ${st['spent']:.2f}; remaining ~${left:.2f} "
          f"(actual/estimate {ratio:.2f}); projected total ${st['spent'] + left:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "status"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--convs", type=Path)
    ap.add_argument("--budget", type=float, help="run: stop before the projected spend passes this many USD")
    ap.add_argument("--spent", type=float, default=0.0, help="run: already spent from this budget before the first slice")
    args = ap.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.cmd == "plan":
        convs = [l.strip() for l in args.convs.read_text(encoding="utf-8").splitlines() if l.strip()]
        cmd_plan(args.run, args.root, convs)
    elif args.cmd == "run":
        cmd_run(args.run, args.root, args.budget, args.spent)
    else:
        cmd_status(args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
