#!/usr/bin/env python3
"""
ora.eval — an evaluation harness for the Ora research engine.

Ora produces open-ended research reports: there is no ground truth to diff against
(unlike Dux's settled trades). So evaluation here follows the three-layer model:

  Layer 1 — output quality : LLM-as-judge against a rubric (skeptical, structured)
  Layer 2 — process health : derived from the run's trace (angles, errors)
  Layer 3 — efficiency     : derived from the trace (calls, tokens, cost, latency)

Two judging modes:

  absolute  — score each report on the rubric. Good for tracking quality over time
              and catching regressions.
  pairwise  — show a judge two reports for the SAME question, blind and with the
              order randomized, and ask which is better. Comparative judgments are
              more reliable than absolute scores, so this is the signal to trust for
              an A/B decision (e.g. "is collapsing the rebuttal seam better, worse,
              or neutral?").

The three commands are decoupled on purpose:

  run     → calls Ora (expensive) and snapshots reports + trace metrics to a result set
  judge   → scores a stored result set (one judge call per question; re-runnable)
  compare → pairwise-judges two stored result sets and prints a verdict

So you pay for Ora once per variant, then judge/compare as much as you like.

CLI:
    python -m ora.eval run --label dedup            # dry-run plan (no spend)
    python -m ora.eval run --label dedup --yes      # actually call Ora
    python -m ora.eval judge --label dedup
    python -m ora.eval compare --a dedup --b collapse
    python -m ora.eval selftest                     # pure-logic checks, no API
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ── Locations ───────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"

# ── Judge model + (approximate) pricing ──────────────────────────────────────

# Judge with a strong model. Ora's deepest stage is already Opus, so to reduce
# self-preference bias prefer a judge that is at least as strong and, ideally,
# a different model family than the thing being judged. Override with --judge-model.
JUDGE_MODEL = "claude-opus-4-7"

# Per-1M-token USD, (input, output). ESTIMATES — update to current rates. The
# eval's value is relative (A vs B, run vs run), so exact prices matter less than
# token counts; cost is a convenience readout, not the primary signal.
PRICING: Dict[str, tuple] = {
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}
CACHE_READ_DISCOUNT = 0.1  # cached input tokens bill at ~10% of the input rate

# ── Rubric (Layer 1) ──────────────────────────────────────────────────────────

# Adapted to what Ora actually is: it reasons from the model's own knowledge with
# no web search, then stress-tests via Critic → rebuttal → synthesis. So the rubric
# rewards calibration and substantiation rather than "sources cited". `calibration`
# is the dimension the Critic/rebuttal seam most affects — it discriminates variants.
@dataclass
class Dimension:
    key: str
    max: int
    desc: str


RUBRIC: List[Dimension] = [
    Dimension("directness", 3, "Directly answers the actual question asked, not an adjacent one."),
    Dimension("comprehensiveness", 3, "Covers the major distinct dimensions the question demands; no glaring blind spot."),
    Dimension("depth", 3, "First-principles reasoning, concrete examples, quantified where possible; not vague generalities."),
    Dimension("calibration", 3, "Distinguishes established fact from speculation; surfaces contested points and uncertainty instead of papering over them."),
    Dimension("coherence", 2, "Well-structured, integrates across angles, internally consistent."),
]
MAX_TOTAL = sum(d.max for d in RUBRIC)


def _rubric_block() -> str:
    return "\n".join(f'  - "{d.key}" (0-{d.max}): {d.desc}' for d in RUBRIC)


# ── Judge prompts ──────────────────────────────────────────────────────────────

ABSOLUTE_SYSTEM = """\
You are a rigorous, skeptical research evaluator. You score research reports against
a fixed rubric. You are hard to impress: top marks mean genuinely excellent, not
merely adequate. Penalize confident claims that aren't substantiated, missing
counter-arguments, and vague generalities. Reward calibrated uncertainty, concrete
specifics, and a direct answer to the question. Return ONLY valid JSON."""

PAIRWISE_SYSTEM = """\
You are a rigorous, skeptical research evaluator. You will see two research reports
answering the SAME question, labeled REPORT 1 and REPORT 2. Decide which better
answers the question, judged on: directness, comprehensiveness, depth and rigor,
calibrated handling of uncertainty, and coherence. Do NOT favor length. Do not
assume either report is correct; reward substantiated, calibrated claims over
confident vagueness. If they are genuinely indistinguishable, say "tie". Return
ONLY valid JSON."""


def _absolute_prompt(question: str, report: str) -> str:
    return (
        f"Original research question:\n{question}\n\n"
        f"Rubric (score each dimension as an integer in its range):\n{_rubric_block()}\n\n"
        f"Report to evaluate:\n---\n{report}\n---\n\n"
        "Return JSON exactly of this shape:\n"
        '{"scores": {<dim_key>: <int>, ...}, '
        '"justifications": {<dim_key>: "<one sentence>", ...}, '
        '"overall": "<2-3 sentence assessment>"}'
    )


def _pairwise_prompt(question: str, r1: str, r2: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"REPORT 1:\n---\n{r1}\n---\n\n"
        f"REPORT 2:\n---\n{r2}\n---\n\n"
        "Return JSON exactly of this shape:\n"
        '{"winner": "1" | "2" | "tie", "margin": "slight" | "clear" | "decisive", '
        '"reason": "<2-3 sentences>"}'
    )


# ── Model client (lazy, injectable) ───────────────────────────────────────────

_client = None


def _get_client():
    """Lazily construct the Anthropic client so importing this module never needs
    an API key (selftest and dry-runs stay key-free)."""
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=600.0, max_retries=4)
    return _client


def _complete(system: str, prompt: str, model: str, max_tokens: int = 1500) -> str:
    """One non-streaming judge call. Indirection point for tests: monkeypatch
    `ora.eval._complete` to evaluate the parsing/aggregation logic without spend."""
    resp = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _extract_json(text: str) -> Optional[dict]:
    """Parse the first JSON object in a model response. Defensive, per Ora's own
    scout-JSON lesson: a judge that wraps JSON in prose shouldn't crash the run."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ── Trace metrics (Layers 2 & 3) ──────────────────────────────────────────────

def _trace_metrics(trace_path: Path) -> dict:
    """Derive process-health and efficiency metrics from an Ora trace JSONL.
    Pure function of the file — unit-testable with a fabricated trace."""
    steps: List[dict] = []
    for line in trace_path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    calls = [s for s in steps if s.get("event") == "model_call"]
    errors = [s for s in steps if s.get("event") == "error"]
    analyst_calls = [c for c in calls if str(c.get("label", "")).startswith("analyst:")]

    tok = {"input": 0, "output": 0, "cached": 0}
    cost = 0.0
    for c in calls:
        t = c.get("tokens", {}) or {}
        for k in tok:
            tok[k] += t.get(k, 0)
        price = PRICING.get(c.get("model", ""))
        if price:
            in_price, out_price = price
            cached = t.get("cached", 0)
            uncached_in = max(t.get("input", 0) - cached, 0)
            cost += (
                uncached_in * in_price
                + cached * in_price * CACHE_READ_DISCOUNT
                + t.get("output", 0) * out_price
            ) / 1_000_000

    starts = [s["ts"] for s in steps if s.get("event") == "run_start"]
    ends = [s["ts"] for s in steps if s.get("event") == "run_end"]
    latency = round(ends[-1] - starts[0], 1) if starts and ends else None

    return {
        "calls": len(calls),
        "tokens": tok,
        "cost_usd": round(cost, 4),
        "errors": len(errors),
        # Layer 2 process health: how many angles got an analyst, and did any fail?
        "angles": len(analyst_calls),
        "analyst_errors": sum(1 for e in errors if str(e.get("label", "")).startswith("analyst:")),
        "scout_fell_back": len(analyst_calls) == 1,  # weak signal of scout JSON-parse fallback
        "latency_s": latency,
    }


# ── Result set model ──────────────────────────────────────────────────────────

@dataclass
class QResult:
    question: str
    slug: str
    report: str = ""
    trace_id: str = ""
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None   # set if the Ora run for this question failed
    judge: Optional[dict] = None  # filled by `judge`


@dataclass
class ResultSet:
    label: str
    depth: str
    created: str
    ora_version: str
    results: List[QResult] = field(default_factory=list)

    def save(self) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{self.label}.json"
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    @staticmethod
    def load(label: str) -> "ResultSet":
        path = RESULTS_DIR / f"{label}.json"
        data = json.loads(path.read_text())
        rs = ResultSet(data["label"], data["depth"], data["created"], data.get("ora_version", "?"))
        rs.results = [QResult(**r) for r in data["results"]]
        return rs


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]


def _ora_version() -> str:
    try:
        from importlib.metadata import version
        return version("ora")
    except Exception:
        return "unknown"


def load_questions() -> List[str]:
    data = json.loads(QUESTIONS_PATH.read_text())
    return [q["question"] if isinstance(q, dict) else q for q in data["questions"]]


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_run(label: str, depth: str, limit: Optional[int], confirm: bool,
            use_memory: bool = False) -> None:
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    approx_calls = len(questions) * 8  # scout + ~N analysts + critic + rebuttal + synth
    print(f"\nrun '{label}'  ·  {len(questions)} question(s)  ·  depth={depth}")
    print(f"  ≈ {approx_calls} model calls against the live API (mostly Opus).")
    if not confirm:
        print("\n  DRY RUN — no API calls made. Re-run with --yes to execute.\n")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q}")
        print()
        return

    import ora.engine as eng

    # Clean-room by default: an eval must be reproducible and its runs independent.
    # Ora persists/recalls findings via Ruflo memory, which would leak state across
    # eval runs (and let earlier runs contaminate later ones). We neutralize the
    # engine's memory functions here rather than change research()'s signature —
    # keeping that contract stable. --use-memory opts back in to evaluate the
    # memory-augmented behavior on purpose.
    if not use_memory:
        eng.memory_store = lambda *a, **k: None
        eng.memory_search = lambda *a, **k: []
        print("  [clean-room] Ruflo memory disabled for run independence "
              "(pass --use-memory to include it)")

    # Eval is batch, not interactive: streaming buys nothing here and is the one
    # failure mode the SDK won't auto-retry — a mid-stream connection reset kills
    # the whole run (observed repeatedly on long Opus phases). Route every phase
    # through Ora's non-streaming runner, which messages.create() + max_retries=4
    # can recover from. research() resolves run_agent as a module global at call
    # time, so reassigning it here redirects scout/critic/rebuttal/synth too.
    eng.run_agent = eng.run_agent_quiet
    print("  [resilient] phases run non-streaming (survives transient connection resets)")
    research = eng.research

    rs = ResultSet(label, depth, datetime.now().isoformat(timespec="seconds"), _ora_version())
    failures = 0
    for i, q in enumerate(questions, 1):
        slug = _slug(q)
        run_dir = RESULTS_DIR / "_runs" / label / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#'*70}\n# [{i}/{len(questions)}] {q}\n{'#'*70}")

        # Isolated output_dir → exactly one trace file per run, no cwd pollution,
        # and runs stay independent of each other.
        try:
            report, err = research(q, depth=depth, save=True, output_dir=run_dir), None
        except Exception as e:
            # Failure isolation: one question's transient blip (e.g. a mid-stream
            # connection reset) must not discard the other questions' completed —
            # and expensive — work. Record it and move on.
            report, err = "", f"{type(e).__name__}: {e}"
            failures += 1
            print(f"\n  ✗ FAILED: {err}")

        traces = sorted((run_dir / "traces").glob("*.jsonl"))
        metrics = _trace_metrics(traces[-1]) if traces else {}
        rs.results.append(QResult(
            question=q, slug=slug, report=report,
            trace_id=traces[-1].stem if traces else "", metrics=metrics, error=err,
        ))
        rs.save()  # checkpoint after every question — a later crash can't erase earlier work

    path = rs.save()
    ok = len(rs.results) - failures
    print(f"\n[saved] {path}  ·  {ok} ok / {failures} failed of {len(rs.results)}")
    _print_efficiency(rs)


def cmd_judge(label: str, model: str) -> None:
    rs = ResultSet.load(label)
    print(f"\njudge '{label}'  ·  {len(rs.results)} report(s)  ·  judge={model}\n")
    judged = []
    for i, r in enumerate(rs.results, 1):
        if not r.report:
            print(f"  [{i}/{len(rs.results)}] —     ·  {r.question[:60]}  (no report{': '+r.error if r.error else ''})")
            continue
        raw = _complete(ABSOLUTE_SYSTEM, _absolute_prompt(r.question, r.report), model)
        parsed = _extract_json(raw) or {}
        scores = parsed.get("scores", {})
        total = sum(int(scores.get(d.key, 0)) for d in RUBRIC)
        r.judge = {"scores": scores, "total": total, "max": MAX_TOTAL,
                   "justifications": parsed.get("justifications", {}),
                   "overall": parsed.get("overall", "")}
        judged.append(total)
        print(f"  [{i}/{len(rs.results)}] {total}/{MAX_TOTAL}  ·  {r.question[:60]}")
    rs.save()
    if judged:
        print(f"\n  mean quality: {sum(judged)/len(judged):.1f}/{MAX_TOTAL}  over {len(judged)} report(s)  (saved into {label}.json)\n")
    else:
        print(f"\n  nothing to judge (no completed reports in '{label}').\n")


def cmd_compare(label_a: str, label_b: str, model: str, repeats: int, seed: int) -> None:
    a, b = ResultSet.load(label_a), ResultSet.load(label_b)
    by_q_b = {r.slug: r for r in b.results}
    rng = random.Random(seed)

    tally = {label_a: 0, label_b: 0, "tie": 0}
    print(f"\ncompare  A='{label_a}'  vs  B='{label_b}'  ·  judge={model}  ·  {repeats}x/question\n")

    for ra in a.results:
        rb = by_q_b.get(ra.slug)
        if rb is None:
            print(f"  (skip — '{ra.slug}' not in B)")
            continue
        if not ra.report or not rb.report:
            print(f"  (skip — missing report for '{ra.slug}')")
            continue
        for _ in range(repeats):
            # Blind + order-randomized: the judge never knows which variant is which,
            # and A isn't always shown first, so position bias can't favor one label.
            a_is_first = rng.random() < 0.5
            r1, r2 = (ra.report, rb.report) if a_is_first else (rb.report, ra.report)
            parsed = _extract_json(_complete(PAIRWISE_SYSTEM, _pairwise_prompt(ra.question, r1, r2), model)) or {}
            winner = parsed.get("winner", "tie")
            if winner == "1":
                victor = label_a if a_is_first else label_b
            elif winner == "2":
                victor = label_b if a_is_first else label_a
            else:
                victor = "tie"
            tally[victor] += 1
        print(f"  · {ra.question[:64]}")

    total = sum(tally.values())
    print(f"\n  ─ verdict ──────────────────────────────────────────")
    print(f"    {label_a:<20} {tally[label_a]:>3}")
    print(f"    {label_b:<20} {tally[label_b]:>3}")
    print(f"    {'tie':<20} {tally['tie']:>3}")
    if total:
        lead = max((label_a, label_b), key=lambda k: tally[k])
        margin = abs(tally[label_a] - tally[label_b])
        if margin == 0:
            print(f"\n    → no measurable difference ({tally['tie']} ties). Treat as neutral.")
        else:
            print(f"\n    → '{lead}' preferred {tally[lead]}/{total} ({margin} net). "
                  f"{'Decisive' if margin > total/2 else 'Suggestive — add questions/repeats to confirm'}.")
    print()


def _print_efficiency(rs: ResultSet) -> None:
    tot_cost = sum(r.metrics.get("cost_usd", 0) for r in rs.results)
    tot_err = sum(r.metrics.get("errors", 0) for r in rs.results)
    print(f"  efficiency: ~${tot_cost:.2f} total · {tot_err} error(s) across {len(rs.results)} run(s)")


# ── Self-test (no API, no spend) ──────────────────────────────────────────────

def cmd_selftest() -> int:
    """Exercise every pure-logic path with fakes. Verifies the harness without a key."""
    import tempfile
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  {'✓' if cond else '✗'} {name}")

    # 1. trace metric extraction from a fabricated trace
    with tempfile.TemporaryDirectory() as d:
        tp = Path(d) / "t.jsonl"
        tp.write_text("\n".join(json.dumps(x) for x in [
            {"event": "run_start", "ts": 100.0, "question": "q"},
            {"event": "model_call", "ts": 101.0, "label": "SCOUT", "model": "claude-sonnet-4-6",
             "tokens": {"input": 1000, "output": 200, "cached": 0}},
            {"event": "model_call", "ts": 102.0, "label": "analyst:A", "model": "claude-opus-4-7",
             "tokens": {"input": 2000, "output": 1000, "cached": 500}},
            {"event": "model_call", "ts": 103.0, "label": "analyst:B", "model": "claude-opus-4-7",
             "tokens": {"input": 2000, "output": 1000, "cached": 0}},
            {"event": "error", "ts": 103.5, "label": "analyst:B", "detail": "boom"},
            {"event": "run_end", "ts": 110.0},
        ]))
        m = _trace_metrics(tp)
        check("trace: counts calls", m["calls"] == 3)
        check("trace: counts angles (analyst calls)", m["angles"] == 2)
        check("trace: counts analyst errors", m["analyst_errors"] == 1)
        check("trace: sums tokens", m["tokens"] == {"input": 5000, "output": 2200, "cached": 500})
        check("trace: computes latency", m["latency_s"] == 10.0)
        check("trace: computes a positive cost", m["cost_usd"] > 0)

    # 2. defensive JSON extraction
    check("json: extracts from prose-wrapped", _extract_json('sure!\n{"winner":"1"} done') == {"winner": "1"})
    check("json: returns None on garbage", _extract_json("no json here") is None)

    # 3. pairwise aggregation with order-randomization (fake judge always prefers
    #    the report containing the marker "GOOD", regardless of slot)
    def fake_complete(system, prompt, model, max_tokens=1500):
        r1 = prompt.split("REPORT 1:")[1].split("REPORT 2:")[0]
        return '{"winner":"1"}' if "GOOD" in r1 else '{"winner":"2"}'

    global _complete
    real = _complete
    _complete = fake_complete
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        a = ResultSet("st_a", "deep", "now", "test")
        a.results = [QResult("q1", "q1", report="GOOD report"), QResult("q2", "q2", report="GOOD two")]
        b = ResultSet("st_b", "deep", "now", "test")
        b.results = [QResult("q1", "q1", report="weak report"), QResult("q2", "q2", report="weak two")]
        a.save(); b.save()
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cmd_compare("st_a", "st_b", "fake", repeats=4, seed=1)
        out = buf.getvalue()
        # 2 questions x 4 repeats = 8 trials. A holds "GOOD" in both reports, so a
        # correct blind/order-randomized tally must credit A with all 8.
        def _tally(label: str) -> str:
            line = next(l for l in out.splitlines() if l.strip().startswith(label))
            return line.strip().split()[-1]
        check("compare: better variant (A) wins all 8 trials", _tally("st_a") == "8")
        check("compare: worse variant (B) wins 0", _tally("st_b") == "0")
    finally:
        _complete = real
        for lbl in ("st_a", "st_b"):
            p = RESULTS_DIR / f"{lbl}.json"
            if p.exists():
                p.unlink()

    print(f"\n{'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(prog="ora-eval", description="Evaluation harness for the Ora research engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run Ora over the eval set into a labeled result set")
    pr.add_argument("--label", required=True, help="name for this result set (e.g. 'dedup', 'collapse')")
    pr.add_argument("--depth", choices=["shallow", "deep"], default="shallow")
    pr.add_argument("--limit", type=int, default=None, help="run only the first N questions")
    pr.add_argument("--yes", action="store_true", help="actually call the API (otherwise dry-run)")
    pr.add_argument("--use-memory", action="store_true",
                    help="include Ruflo memory (default: clean-room, memory disabled for run independence)")

    pj = sub.add_parser("judge", help="score a stored result set on the rubric (LLM-as-judge)")
    pj.add_argument("--label", required=True)
    pj.add_argument("--judge-model", default=JUDGE_MODEL)

    pc = sub.add_parser("compare", help="blind pairwise A/B between two stored result sets")
    pc.add_argument("--a", required=True)
    pc.add_argument("--b", required=True)
    pc.add_argument("--judge-model", default=JUDGE_MODEL)
    pc.add_argument("--repeats", type=int, default=1, help="judge passes per question")
    pc.add_argument("--seed", type=int, default=0)

    sub.add_parser("selftest", help="run pure-logic checks (no API, no spend)")

    args = p.parse_args()
    if args.cmd == "run":
        cmd_run(args.label, args.depth, args.limit, args.yes, args.use_memory)
    elif args.cmd == "judge":
        cmd_judge(args.label, args.judge_model)
    elif args.cmd == "compare":
        cmd_compare(args.a, args.b, args.judge_model, args.repeats, args.seed)
    elif args.cmd == "selftest":
        sys.exit(cmd_selftest())


if __name__ == "__main__":
    main()
