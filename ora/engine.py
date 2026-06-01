#!/usr/bin/env python3
"""
ora — Multi-Agent Deep Research Engine

Four agents run in sequence (collapse variant — no separate rebuttal stage):
  Scout         — decomposes the question into distinct research angles
  Analyst Swarm — each analyst dives deep on one angle
  Critic        — stress-tests findings, surfaces gaps and contradictions
  Synthesizer   — weighs the critique and merges everything into a report

Findings are stored in Ruflo memory so future sessions build on prior research.
Reports are saved to ./reports/ relative to the working directory.

CLI:
    ora "What are the systemic risks of AI in financial markets?"
    ora --depth shallow "How does mRNA therapy work?"
    ora --no-save "What caused the 2008 financial crisis?"

Module:
    from ora import research
    report = research("What is X?", depth="deep")
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

import os

import anthropic

# ── Optional web search (Tavily) ──────────────────────────────────────────────
# Install: pip install tavily-python
# Enable:  export TAVILY_API_KEY=tvly-...
try:
    from tavily import TavilyClient as _TavilyClient
    _tavily: "_TavilyClient | None" = (
        _TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        if "TAVILY_API_KEY" in os.environ
        else None
    )
except ImportError:
    _tavily = None

# ── Models ────────────────────────────────────────────────────────────────────

# max_retries lets the SDK auto-retry transient faults (timeouts, 429, 5xx) with
# backoff — no need to hand-roll a retry loop. timeout is generous for the long,
# non-streaming deep calls the parallel analyst swarm makes.
client = anthropic.Anthropic(timeout=1200.0, max_retries=4)

MODEL_FAST = "claude-sonnet-4-6"  # Scout, Critic — analytical, low-latency
MODEL_DEEP = "claude-opus-4-7"    # Analyst, Rebuttal, Synthesizer — maximum depth

DEPTH_TOKENS = {"shallow": 2048, "deep": 6144}

# ── System prompts ────────────────────────────────────────────────────────────

SCOUT_SYSTEM = """\
You are a research scout. Given a question, identify 3–5 distinct research angles
that together give a complete picture. Angles must be orthogonal — each covering
a different dimension (e.g. technical, economic, historical, ethical, practical).

Return a JSON array of objects with exactly these keys:
  "angle"    — short name, 3–6 words
  "question" — specific sub-question to investigate
  "priority" — "high" | "medium"

Return ONLY valid JSON. No prose, no markdown fences."""

ANALYST_SYSTEM = """\
You are a deep research analyst. Given a research angle and sub-question, conduct
thorough, rigorous analysis. Ground findings in first principles, name concrete
examples, quantify where possible, and distinguish established fact from
speculation. Use clear markdown headers to structure your output.

Be comprehensive and precise. Avoid vague generalities."""

CRITIC_SYSTEM = """\
You are a critical reviewer. Stress-test analyst findings by identifying:
  1. Logical gaps or non-sequiturs
  2. Missing counterarguments or alternative explanations
  3. Unsupported or overconfident claims
  4. Important angles missed entirely
  5. Contradictions between different analysts' findings

For each issue: quote the specific claim, name the problem, explain why it matters.
Do not rewrite findings — only challenge them. Structure output as markdown."""

SYNTHESIZER_SYSTEM = """\
You are a senior research synthesizer. Given multi-analyst findings that have been
stress-tested by a critic, write a comprehensive report — weighing the critic's
challenges directly, correcting or qualifying findings where a challenge holds and
pushing back where it does not:

# [Descriptive Title]

## Executive Summary
3–5 sentences capturing the core answer and key tensions.

## Key Findings
One subsection per major theme, integrating across all research angles.

## Uncertainties & Open Questions
What remains contested, unknown, or highly context-dependent.

## Conclusion
Direct answer to the original question, plus key implications.

---
Write for an intelligent non-specialist. Be direct. Where issues remain unresolved
after the critique, say so plainly rather than papering over them."""


# ── Web search helpers ────────────────────────────────────────────────────────

def _web_search(query: str, max_results: int = 5) -> List[dict]:
    """Return up to *max_results* results as {title, url, content} dicts.
    Returns [] silently when Tavily is unavailable or the call fails."""
    if _tavily is None:
        return []
    try:
        resp = _tavily.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in resp.get("results", [])
        ]
    except Exception:
        return []


def _format_sources(results: List[dict]) -> str:
    if not results:
        return ""
    lines = ["## Web Sources\n"]
    for i, r in enumerate(results, 1):
        snippet = r["content"][:800].rstrip()
        lines.append(f"**[{i}] {r['title']}**  \n{r['url']}\n\n{snippet}\n")
    return "\n".join(lines)


# ── Memory helpers (Ruflo CLI) ────────────────────────────────────────────────

def _ruflo(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["npx", "@claude-flow/cli@latest", *args],
        capture_output=True, text=True, timeout=20, cwd=cwd,
    )


def memory_store(key: str, value: str, tags: List[str], cwd: Path) -> None:
    try:
        _ruflo("memory", "store", "-k", key, "--value", value,
               "--tags", ",".join(tags), cwd=cwd)
    except Exception:
        pass


def memory_search(query: str, cwd: Path) -> List[dict]:
    try:
        result = _ruflo("memory", "search", "--query", query, "--limit", "3", cwd=cwd)
        if result.returncode == 0 and result.stdout.strip():
            memories = []
            for line in result.stdout.strip().splitlines():
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return memories
    except Exception:
        pass
    return []


# ── Observability ─────────────────────────────────────────────────────────────

class Trace:
    """Structured per-run trace written to ./traces/<id>.jsonl. The trace is the
    unit of observability: it records every model call (model, tokens) and phase
    boundary, so a finished run is debuggable and its cost is visible after the
    fact — Ora previously logged nothing structured."""

    def __init__(self, cwd: Path):
        self.id = uuid.uuid4().hex[:12]
        self.steps: List[dict] = []
        traces_dir = cwd / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        self.path = traces_dir / f"{self.id}.jsonl"

    def log(self, event: str, **fields: Any) -> None:
        record = {"trace_id": self.id, "ts": time.time(), "event": event, **fields}
        self.steps.append(record)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def totals(self) -> dict:
        tok = {"input": 0, "output": 0, "cached": 0}
        for s in self.steps:
            for k in tok:
                tok[k] += s.get("tokens", {}).get(k, 0)
        return {
            "trace_id": self.id,
            "calls": sum(1 for s in self.steps if s["event"] == "model_call"),
            "tokens": tok,
            "errors": sum(1 for s in self.steps if s["event"] == "error"),
        }


def _usage_tokens(obj: Any) -> dict:
    u = getattr(obj, "usage", None)
    return {
        "input": getattr(u, "input_tokens", 0),
        "output": getattr(u, "output_tokens", 0),
        "cached": getattr(u, "cache_read_input_tokens", 0),
    }


# ── Agent runners ───────────────────────────────────────────────────────────

def run_agent(
    system: str,
    prompt: str,
    model: str,
    max_tokens: int,
    label: str = "",
    thinking: bool = False,
    trace: "Trace | None" = None,
) -> str:
    """Streaming runner — used for the single-call phases (scout, critic, synth)
    where live token output is good UX."""
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}\n")

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    collected: List[str] = []
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            collected.append(text)
        final = stream.get_final_message()

    print()
    if trace is not None:
        trace.log("model_call", label=label or "agent", model=model,
                  streamed=True, tokens=_usage_tokens(final))
    return "".join(collected)


def run_agent_quiet(
    system: str,
    prompt: str,
    model: str,
    max_tokens: int,
    label: str = "",
    thinking: bool = False,
    trace: "Trace | None" = None,
) -> str:
    """Non-streaming runner — used for the parallel analyst swarm, where
    concurrent token streams would scramble the terminal. Returns the full text."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    resp = client.messages.create(**kwargs)
    if trace is not None:
        trace.log("model_call", label=label or "agent", model=model,
                  streamed=False, tokens=_usage_tokens(resp))
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ResearchAngle:
    angle: str
    question: str
    priority: str
    findings: str = ""


# ── Research phases ───────────────────────────────────────────────────────────

def scout_phase(question: str, trace: "Trace | None" = None) -> List[ResearchAngle]:
    _banner("SCOUT", "Mapping the research landscape")

    raw = run_agent(
        system=SCOUT_SYSTEM,
        prompt=f"Question to research: {question}",
        model=MODEL_FAST,
        max_tokens=1024,
        label="SCOUT  ·  Decomposing into angles",
        trace=trace,
    )

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        return [ResearchAngle(angle="Full Analysis", question=question, priority="high")]

    try:
        data = json.loads(json_match.group())
        return [
            ResearchAngle(
                angle=item.get("angle", f"Angle {i + 1}"),
                question=item.get("question", question),
                priority=item.get("priority", "high"),
            )
            for i, item in enumerate(data)
        ]
    except (json.JSONDecodeError, AttributeError):
        return [ResearchAngle(angle="Full Analysis", question=question, priority="high")]


def analyst_phase(
    question: str, angles: List[ResearchAngle], max_tokens: int,
    web: bool = True, trace: "Trace | None" = None
) -> None:
    """Run the analyst swarm in parallel. The angles are independent, so they
    fan out concurrently (was sequential) — wall-clock time drops to ~the slowest
    analyst instead of the sum. Non-streaming calls (run_agent_quiet) avoid
    interleaved terminal output; failures are isolated per analyst."""
    _banner("ANALYST SWARM", f"{len(angles)} parallel research threads")

    def analyze(item: "tuple[int, ResearchAngle]") -> None:
        i, angle = item

        # Fetch live web context before the analyst writes their findings.
        # Sources are injected into the user prompt so the system prompt stays
        # stable (preserving the cache_control ephemeral hit across analysts).
        sources_block = ""
        if web and _tavily is not None:
            results = _web_search(angle.question)
            if trace is not None:
                trace.log("web_search", angle=angle.angle, query=angle.question,
                          results=len(results))
            if results:
                sources_block = (
                    "\n\n" + _format_sources(results) +
                    "\nCite these sources inline as [1], [2], etc. where relevant."
                )

        angle.findings = run_agent_quiet(
            system=ANALYST_SYSTEM,
            prompt=(
                f"Original question: {question}\n\n"
                f"Your research angle: {angle.angle}\n"
                f"Specific sub-question: {angle.question}"
                f"{sources_block}"
            ),
            model=MODEL_DEEP,
            max_tokens=max_tokens,
            label=f"analyst:{angle.angle}",
            thinking=True,
            trace=trace,
        )

    items = list(enumerate(angles, 1))
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        futures = {ex.submit(analyze, it): it for it in items}
        for fut in as_completed(futures):
            i, angle = futures[fut]
            try:
                fut.result()
                print(f"  ✓  ANALYST {i}/{len(angles)}  ·  {angle.angle}  ({len(angle.findings)} chars)")
            except Exception as e:  # isolation — one analyst failing doesn't kill the swarm
                if trace is not None:
                    trace.log("error", layer=1, label=f"analyst:{angle.angle}", detail=str(e))
                print(f"  ✗  ANALYST {i}/{len(angles)}  ·  {angle.angle}  FAILED: {e}")

    # A systematic failure (every analyst down) must not pass silently as "no findings".
    if not any(a.findings for a in angles):
        print("\n  [warning] all analysts failed — downstream phases have no findings to work from.")


def critic_phase(question: str, angles: List[ResearchAngle], trace: "Trace | None" = None) -> str:
    _banner("CRITIC", "Stress-testing findings")

    combined = "\n\n".join(f"### {a.angle}\n{a.findings}" for a in angles)

    return run_agent(
        system=CRITIC_SYSTEM,
        prompt=(
            f"Original question: {question}\n\n"
            f"Analyst findings to review:\n\n{combined}"
        ),
        model=MODEL_FAST,
        max_tokens=3072,
        label="CRITIC  ·  Identifying gaps and contradictions",
        trace=trace,
    )


def synthesis_phase(
    question: str, angles: List[ResearchAngle], critique: str,
    trace: "Trace | None" = None
) -> str:
    """Collapse variant: the rebuttal/consensus seam is removed. The synthesizer
    receives the analyst findings and the critic's challenges directly and is
    instructed to weigh and resolve the critique itself, rather than relying on a
    separate Opus rebuttal stage to pre-digest it."""
    _banner("SYNTHESIZER", "Writing final report")

    findings = "\n\n---\n\n".join(
        f"## Angle: {a.angle}\n\n{a.findings}" for a in angles
    )
    context = (
        f"Research question: {question}\n\n"
        f"{findings}\n\n"
        f"---\n\n## Critic's challenges (weigh and resolve these as you synthesize):\n{critique}"
    )

    return run_agent(
        system=SYNTHESIZER_SYSTEM,
        prompt=context,
        model=MODEL_DEEP,
        max_tokens=6144,
        label="SYNTHESIZER  ·  Final report",
        thinking=True,
        trace=trace,
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _banner(title: str, subtitle: str = "") -> None:
    print(f"\n{'═' * 60}")
    suffix = f"  —  {subtitle}" if subtitle else ""
    print(f"  {title}{suffix}")
    print(f"{'═' * 60}")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


# ── Public API ────────────────────────────────────────────────────────────────

def research(
    question: str,
    depth: str = "deep",
    save: bool = True,
    output_dir: "Path | None" = None,
    web: bool = True,
) -> str:
    """Run a full multi-agent research session on *question*.

    Args:
        question:   The research question.
        depth:      "shallow" (faster) or "deep" (thorough, default).
        save:       Write the report to *output_dir*/reports/ when True.
        output_dir: Base directory for reports and Ruflo memory calls.
                    Defaults to the current working directory.
        web:        Enable live web search via Tavily (requires TAVILY_API_KEY).
                    Falls back silently to knowledge-only mode if unavailable.

    Returns:
        The final synthesized report as a markdown string.
    """
    cwd = Path(output_dir) if output_dir else Path.cwd()
    max_tokens = DEPTH_TOKENS[depth]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    trace = Trace(cwd)
    web_active = web and _tavily is not None
    trace.log("run_start", question=question, depth=depth, web=web_active)

    _banner("ORA  ·  Deep Research Engine", ts)
    print(f"\n  Question : {question}")
    print(f"  Depth    : {depth}  ({max_tokens} tokens/analyst)")
    web_status = "enabled" if web_active else ("disabled (--no-web)" if not web else "unavailable (set TAVILY_API_KEY)")
    print(f"  Web      : {web_status}")
    print(f"  Trace    : traces/{trace.id}.jsonl")

    prior = memory_search(question, cwd=cwd)
    if prior:
        print(f"\n  [memory] {len(prior)} related session(s) found")

    angles = scout_phase(question, trace=trace)
    print(f"\n  Angles identified:")
    for a in angles:
        marker = "●" if a.priority == "high" else "○"
        print(f"    {marker}  {a.angle}: {a.question}")

    analyst_phase(question, angles, max_tokens, web=web, trace=trace)
    critique = critic_phase(question, angles, trace=trace)
    report = synthesis_phase(question, angles, critique, trace=trace)
    trace.log("run_end")

    # Persist to Ruflo memory
    angle_tags = [_slug(a.angle) for a in angles]
    memory_store(
        key=f"research/{_slug(question)}",
        value=report[:2000],
        tags=["research", "report", depth] + angle_tags,
        cwd=cwd,
    )

    if save:
        reports_dir = cwd / "reports"
        reports_dir.mkdir(exist_ok=True)
        filename = (
            reports_dir
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_slug(question)[:40]}.md"
        )
        filename.write_text(
            f"# Research Report\n\n"
            f"**Question:** {question}  \n"
            f"**Date:** {ts}  \n"
            f"**Depth:** {depth}  \n\n"
            f"---\n\n{report}"
        )
        print(f"\n  [saved] reports/{filename.name}")

    totals = trace.totals()
    tok = totals["tokens"]
    searches = sum(1 for s in trace.steps if s["event"] == "web_search")
    search_note = f" · {searches} web search(es)" if searches else ""
    print(f"\n  [trace] {totals['calls']} calls{search_note} · "
          f"{tok['input']} in / {tok['output']} out / {tok['cached']} cached tokens · "
          f"{totals['errors']} error(s) · traces/{trace.id}.jsonl")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ora",
        description="Multi-Agent Deep Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              ora "What are the systemic risks of AI in financial markets?"
              ora --depth shallow "How does CRISPR base editing work?"
              ora --no-save "What caused the 2008 financial crisis?"
        """),
    )
    parser.add_argument("question", nargs="+", help="The research question")
    parser.add_argument(
        "--depth",
        choices=["shallow", "deep"],
        default="deep",
        help="Research depth — shallow is faster, deep is thorough (default: deep)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the report to ./reports/",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable live web search (use model knowledge only)",
    )
    args = parser.parse_args()

    research(" ".join(args.question), depth=args.depth, save=not args.no_save,
             web=not args.no_web)

    _banner("COMPLETE")
    print()


if __name__ == "__main__":
    main()
