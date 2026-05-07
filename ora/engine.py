#!/usr/bin/env python3
"""
ora — Multi-Agent Deep Research Engine

Five agents run in sequence:
  Scout         — decomposes the question into distinct research angles
  Analyst Swarm — each analyst dives deep on one angle
  Critic        — stress-tests findings, surfaces gaps and contradictions
  Consensus     — analysts respond to critique and refine findings
  Synthesizer   — merges everything into a structured markdown report

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

import anthropic

# ── Models ────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

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

REBUTTAL_SYSTEM = """\
You are a research analyst responding to a critical review of your team's findings.
For each challenge raised by the critic:
  - If valid: acknowledge it and correct or qualify the finding
  - If partially valid: incorporate the nuance
  - If invalid: explain specifically why, with evidence or reasoning

Be precise and brief. Return only updated or annotated findings."""

SYNTHESIZER_SYSTEM = """\
You are a senior research synthesizer. Given multi-analyst findings that have been
stress-tested by a critic and refined via rebuttal, write a comprehensive report:

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


# ── Streaming agent runner ────────────────────────────────────────────────────

def run_agent(
    system: str,
    prompt: str,
    model: str,
    max_tokens: int,
    label: str = "",
    thinking: bool = False,
) -> str:
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

    print()
    return "".join(collected)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ResearchAngle:
    angle: str
    question: str
    priority: str
    findings: str = ""
    refined_findings: str = ""


# ── Research phases ───────────────────────────────────────────────────────────

def scout_phase(question: str) -> List[ResearchAngle]:
    _banner("SCOUT", "Mapping the research landscape")

    raw = run_agent(
        system=SCOUT_SYSTEM,
        prompt=f"Question to research: {question}",
        model=MODEL_FAST,
        max_tokens=1024,
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


def analyst_phase(question: str, angles: List[ResearchAngle], max_tokens: int) -> None:
    _banner("ANALYST SWARM", f"{len(angles)} research threads")

    for i, angle in enumerate(angles, 1):
        angle.findings = run_agent(
            system=ANALYST_SYSTEM,
            prompt=(
                f"Original question: {question}\n\n"
                f"Your research angle: {angle.angle}\n"
                f"Specific sub-question: {angle.question}"
            ),
            model=MODEL_DEEP,
            max_tokens=max_tokens,
            label=f"ANALYST {i}/{len(angles)}  ·  {angle.angle}",
            thinking=True,
        )


def critic_phase(question: str, angles: List[ResearchAngle]) -> str:
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
    )


def consensus_phase(question: str, angles: List[ResearchAngle], critique: str) -> None:
    _banner("CONSENSUS", "Analysts respond to critique")

    combined = "\n\n".join(f"### {a.angle}\n{a.findings}" for a in angles)

    refined = run_agent(
        system=REBUTTAL_SYSTEM,
        prompt=(
            f"Original question: {question}\n\n"
            f"Analyst findings:\n{combined}\n\n"
            f"Critic's challenges:\n{critique}"
        ),
        model=MODEL_DEEP,
        max_tokens=4096,
        label="REBUTTAL  ·  Incorporating critique",
        thinking=True,
    )

    for angle in angles:
        angle.refined_findings = refined


def synthesis_phase(question: str, angles: List[ResearchAngle], critique: str) -> str:
    _banner("SYNTHESIZER", "Writing final report")

    context = (
        f"Research question: {question}\n\n"
        + "\n\n---\n\n".join(
            f"## Angle: {a.angle}\n\n"
            f"**Original findings:**\n{a.findings}\n\n"
            f"**Refined post-critique:**\n{a.refined_findings or '(no changes needed)'}"
            for a in angles
        )
        + f"\n\n---\n\n## Critic's unresolved challenges:\n{critique}"
    )

    return run_agent(
        system=SYNTHESIZER_SYSTEM,
        prompt=context,
        model=MODEL_DEEP,
        max_tokens=6144,
        label="SYNTHESIZER  ·  Final report",
        thinking=True,
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
) -> str:
    """Run a full multi-agent research session on *question*.

    Args:
        question:   The research question.
        depth:      "shallow" (faster) or "deep" (thorough, default).
        save:       Write the report to *output_dir*/reports/ when True.
        output_dir: Base directory for reports and Ruflo memory calls.
                    Defaults to the current working directory.

    Returns:
        The final synthesized report as a markdown string.
    """
    cwd = Path(output_dir) if output_dir else Path.cwd()
    max_tokens = DEPTH_TOKENS[depth]
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    _banner("ORA  ·  Deep Research Engine", ts)
    print(f"\n  Question : {question}")
    print(f"  Depth    : {depth}  ({max_tokens} tokens/analyst)")

    prior = memory_search(question, cwd=cwd)
    if prior:
        print(f"\n  [memory] {len(prior)} related session(s) found")

    angles = scout_phase(question)
    print(f"\n  Angles identified:")
    for a in angles:
        marker = "●" if a.priority == "high" else "○"
        print(f"    {marker}  {a.angle}: {a.question}")

    analyst_phase(question, angles, max_tokens)
    critique = critic_phase(question, angles)
    consensus_phase(question, angles, critique)
    report = synthesis_phase(question, angles, critique)

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
    args = parser.parse_args()

    research(" ".join(args.question), depth=args.depth, save=not args.no_save)

    _banner("COMPLETE")
    print()


if __name__ == "__main__":
    main()
