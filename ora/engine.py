#!/usr/bin/env python3
"""
ora — Multi-Agent Deep Research Engine

Five agents run in sequence:
  Scout         — decomposes the question into distinct research angles
  Analyst Swarm — each analyst dives deep on one angle (with live web sources)
  Compress      — distils each analyst report to key findings (Haiku, parallel)
  Critic        — stress-tests compressed findings, surfaces gaps
  Synthesizer   — weighs the critique and merges everything into a report

Model routing (3-tier):
  shallow → Scout:Haiku  Analyst:Sonnet  Compress:Haiku  Critic:Haiku  Synth:Sonnet
  deep    → Scout:Haiku  Analyst:Opus    Compress:Haiku  Critic:Sonnet Synth:Opus

CLI:
    ora "What are the systemic risks of AI in financial markets?"
    ora --depth shallow "How does mRNA therapy work?"
    ora --no-save "What caused the 2008 financial crisis?"
    ora --no-web "What is the philosophy of Stoicism?"

Module:
    from ora import research
    report = research("What is X?", depth="deep")
"""
from __future__ import annotations

import argparse
import json
import os
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

# ── Models & routing ──────────────────────────────────────────────────────────

client = anthropic.Anthropic(timeout=1200.0, max_retries=4)

MODEL_HAIKU = "claude-haiku-4-5-20251001"   # tier 2 — fast, cheap
MODEL_FAST  = "claude-sonnet-4-6"            # tier 3 — balanced
MODEL_DEEP  = "claude-opus-4-7"              # tier 4 — maximum depth

DEPTH_TOKENS = {"shallow": 2048, "deep": 6144}


def _route_models(depth: str) -> dict:
    """3-tier model routing — cheap models for simple tasks, Opus only where needed."""
    if depth == "shallow":
        return dict(scout=MODEL_HAIKU, analyst=MODEL_FAST, compress=MODEL_HAIKU,
                    critic=MODEL_HAIKU, synthesizer=MODEL_FAST)
    return dict(scout=MODEL_HAIKU, analyst=MODEL_DEEP, compress=MODEL_HAIKU,
                critic=MODEL_FAST, synthesizer=MODEL_DEEP)


def _model_label(m: str) -> str:
    for name in ("haiku", "sonnet", "opus"):
        if name in m:
            return name
    return m


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
You are a deep research analyst. Your job is to produce a thorough, rigorous
investigation of a specific research angle within a broader question.

When web sources are provided, ground your analysis in those sources first.
When no sources are provided, ground analysis in first principles and established
knowledge. In both cases: name concrete examples, quantify where possible, and
clearly distinguish established fact from inference or speculation.

Structure your output with clear markdown headers. Your analysis should cover:

1. **Core findings** — what is actually known or demonstrably true about this angle,
   with specific evidence, numbers, and named examples

2. **Mechanisms and explanations** — the underlying reasons, causes, or dynamics
   that explain the findings; not just what but why

3. **Counterarguments and limits** — where the evidence is weak, contested, or
   context-dependent; what a reasonable skeptic would object to

4. **Implications** — what the findings on this angle mean for the broader question

Precision over breadth. A single well-evidenced claim is worth more than five
vague assertions. Cite sources inline as [1], [2] when web results are provided."""

COMPRESS_SYSTEM = """\
You are a research distiller. Compress a detailed analyst report into a tight,
information-dense summary that preserves every specific claim, number, named
source, and cited URL — while eliminating repetition and verbose explanation.

Return structured markdown with exactly these sections:

## Key Findings
3–6 bullet points. Each must be specific and concrete: name the claim, the
evidence or source, and the magnitude or confidence level where available.

## Sources & Evidence
1–3 sentences naming the strongest sources cited, with URLs if present.
If no web sources were cited, state that explicitly.

## Primary Uncertainty
1 sentence identifying the main unresolved question or caveat in this angle.

Maximum 400 words. Do not add new claims. Do not explain your process."""

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
    """Return up to *max_results* results as {title, url, content} dicts."""
    if _tavily is None:
        return []
    try:
        resp = _tavily.search(query, max_results=max_results)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in resp.get("results", [])
        ]
    except Exception:
        return []


def _format_sources(results: List[dict]) -> str:
    if not results:
        return ""
    lines = ["## Web Sources\n"]
    for i, r in enumerate(results, 1):
        snippet = r["content"][:350].rstrip()  # tightened: 800→350 chars
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
    """Structured per-run trace written to ./traces/<id>.jsonl."""

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


# ── Agent runners ─────────────────────────────────────────────────────────────

def run_agent(
    system: str,
    prompt: str,
    model: str,
    max_tokens: int,
    label: str = "",
    thinking: bool = False,
    trace: "Trace | None" = None,
    messages: "List[dict] | None" = None,
) -> str:
    """Streaming runner for single-call phases (scout, critic, synthesizer)."""
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}\n")

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages or [{"role": "user", "content": prompt}],
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
    messages: "List[dict] | None" = None,
) -> str:
    """Non-streaming runner for parallel swarms (analysts, compress)."""
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages or [{"role": "user", "content": prompt}],
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
    compressed: str = ""  # distilled findings — used by Critic and Synthesizer


# ── Research phases ───────────────────────────────────────────────────────────

def scout_phase(
    question: str, models: dict, trace: "Trace | None" = None
) -> List[ResearchAngle]:
    _banner("SCOUT", f"Mapping the research landscape  [{_model_label(models['scout'])}]")

    raw = run_agent(
        system=SCOUT_SYSTEM,
        prompt=f"Question to research: {question}",
        model=models["scout"],
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
    models: dict, web: bool = True, trace: "Trace | None" = None
) -> None:
    """Run the analyst swarm in parallel."""
    _banner("ANALYST SWARM",
            f"{len(angles)} parallel threads  [{_model_label(models['analyst'])}]")

    def analyze(item: "tuple[int, ResearchAngle]") -> None:
        i, angle = item
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
            model=models["analyst"],
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
            except Exception as e:
                if trace is not None:
                    trace.log("error", layer="analyst", label=angle.angle, detail=str(e))
                print(f"  ✗  ANALYST {i}/{len(angles)}  ·  {angle.angle}  FAILED: {e}")

    if not any(a.findings for a in angles):
        print("\n  [warning] all analysts failed — downstream phases have no findings.")


def compress_phase(
    angles: List[ResearchAngle], models: dict, trace: "Trace | None" = None
) -> None:
    """Compress each analyst's findings in parallel using Haiku.
    Populates angle.compressed — passed to Critic and Synthesizer instead of
    full findings, cutting their input tokens by ~70-80%. Falls back to full
    findings if compression fails."""
    _banner("COMPRESS",
            f"Distilling {len(angles)} reports  [{_model_label(models['compress'])}]")

    def compress_one(item: "tuple[int, ResearchAngle]") -> None:
        i, angle = item
        if not angle.findings:
            angle.compressed = ""
            return
        try:
            angle.compressed = run_agent_quiet(
                system=COMPRESS_SYSTEM,
                prompt=f"## {angle.angle}\n\n{angle.findings}",
                model=models["compress"],
                max_tokens=600,
                label=f"compress:{angle.angle}",
                thinking=False,
                trace=trace,
            )
        except Exception as e:
            if trace is not None:
                trace.log("error", layer="compress", label=angle.angle, detail=str(e))
            angle.compressed = angle.findings  # fallback to full findings

    items = list(enumerate(angles, 1))
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        futures = {ex.submit(compress_one, it): it for it in items}
        for fut in as_completed(futures):
            i, angle = futures[fut]
            try:
                fut.result()
                if angle.findings:
                    pct = int(100 * len(angle.compressed) / len(angle.findings))
                    print(f"  ✓  COMPRESS {i}/{len(angles)}  ·  {angle.angle}  "
                          f"({len(angle.compressed)}/{len(angle.findings)} chars, {pct}%)")
                else:
                    print(f"  –  COMPRESS {i}/{len(angles)}  ·  {angle.angle}  skipped (no findings)")
            except Exception as e:
                print(f"  ✗  COMPRESS {i}/{len(angles)}  ·  {angle.angle}  FAILED: {e}")


def critic_phase(
    question: str, angles: List[ResearchAngle], models: dict,
    trace: "Trace | None" = None
) -> str:
    _banner("CRITIC",
            f"Stress-testing findings  [{_model_label(models['critic'])}]")

    # Use compressed findings (with cache_control) — Synthesizer receives the
    # same block and gets a prompt-cache hit, saving the full findings cost again.
    combined = "\n\n".join(
        f"### {a.angle}\n{a.compressed or a.findings}" for a in angles
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": combined, "cache_control": {"type": "ephemeral"}},
            {"type": "text",
             "text": f"\nOriginal question: {question}\n\nStress-test these findings."},
        ],
    }]

    return run_agent(
        system=CRITIC_SYSTEM,
        prompt="",
        model=models["critic"],
        max_tokens=3072,
        label="CRITIC  ·  Identifying gaps and contradictions",
        trace=trace,
        messages=messages,
    )


def synthesis_phase(
    question: str, angles: List[ResearchAngle], critique: str, models: dict,
    trace: "Trace | None" = None
) -> str:
    _banner("SYNTHESIZER",
            f"Writing final report  [{_model_label(models['synthesizer'])}]")

    # Same combined block as Critic — gets a prompt-cache hit on the findings.
    combined = "\n\n".join(
        f"### {a.angle}\n{a.compressed or a.findings}" for a in angles
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": combined, "cache_control": {"type": "ephemeral"}},
            {"type": "text",
             "text": (
                 f"\nResearch question: {question}\n\n"
                 f"---\n\n## Critic's challenges (weigh and resolve these):\n{critique}"
             )},
        ],
    }]

    return run_agent(
        system=SYNTHESIZER_SYSTEM,
        prompt="",
        model=models["synthesizer"],
        max_tokens=6144,
        label="SYNTHESIZER  ·  Final report",
        thinking=True,
        trace=trace,
        messages=messages,
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
        depth:      "shallow" (faster/cheaper) or "deep" (thorough, default).
        save:       Write the report to *output_dir*/reports/ when True.
        output_dir: Base directory for reports and Ruflo memory calls.
        web:        Enable live web search via Tavily (requires TAVILY_API_KEY).

    Returns:
        The final synthesized report as a markdown string.
    """
    cwd = Path(output_dir) if output_dir else Path.cwd()
    max_tokens = DEPTH_TOKENS[depth]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    trace = Trace(cwd)
    models = _route_models(depth)
    web_active = web and _tavily is not None
    trace.log("run_start", question=question, depth=depth, web=web_active, models=models)

    _banner("ORA  ·  Deep Research Engine", ts)
    print(f"\n  Question : {question}")
    print(f"  Depth    : {depth}  ({max_tokens} tokens/analyst)")
    web_status = ("enabled" if web_active
                  else ("disabled (--no-web)" if not web
                        else "unavailable (set TAVILY_API_KEY)"))
    print(f"  Web      : {web_status}")
    print(f"  Models   : scout={_model_label(models['scout'])}  "
          f"analyst={_model_label(models['analyst'])}  "
          f"critic={_model_label(models['critic'])}  "
          f"synth={_model_label(models['synthesizer'])}")
    print(f"  Trace    : traces/{trace.id}.jsonl")

    prior = memory_search(question, cwd=cwd)
    if prior:
        print(f"\n  [memory] {len(prior)} related session(s) found")

    angles = scout_phase(question, models=models, trace=trace)
    print(f"\n  Angles identified:")
    for a in angles:
        marker = "●" if a.priority == "high" else "○"
        print(f"    {marker}  {a.angle}: {a.question}")

    analyst_phase(question, angles, max_tokens, models=models, web=web, trace=trace)
    compress_phase(angles, models=models, trace=trace)
    critique = critic_phase(question, angles, models=models, trace=trace)
    report = synthesis_phase(question, angles, critique, models=models, trace=trace)
    trace.log("run_end")

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
    search_note = f" · {searches} search(es)" if searches else ""
    cached_note = f" · {tok['cached']} cached" if tok["cached"] else ""
    print(f"\n  [trace] {totals['calls']} calls{search_note} · "
          f"{tok['input']} in / {tok['output']} out{cached_note} tokens · "
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
              ora --no-web "What is the philosophy of Stoicism?"
        """),
    )
    parser.add_argument("question", nargs="+", help="The research question")
    parser.add_argument(
        "--depth",
        choices=["shallow", "deep"],
        default="deep",
        help="Research depth — shallow is faster/cheaper, deep is thorough (default: deep)",
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
