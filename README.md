# Ora — Multi-Agent Deep Research Engine

Ora turns a single question into a comprehensive research report by running four specialized AI agents in sequence. Each agent has a distinct role, a tuned system prompt, and the right model for its job — the result is a report that has been challenged and defended before you read it.

```
Scout → Analyst Swarm → Critic → Synthesizer
```

---

## How it works

| Agent | Model | Job |
|---|---|---|
| **Scout** | Sonnet 4.6 | Breaks the question into 3–5 orthogonal research angles |
| **Analyst Swarm** | Opus 4.7 + thinking | Each analyst dives deep on one angle, grounded in live web sources |
| **Critic** | Sonnet 4.6 | Stress-tests all findings — gaps, contradictions, unsupported claims |
| **Synthesizer** | Opus 4.7 + thinking | Weighs the critique directly and merges everything into a structured report |

Every session is stored in [Ruflo](https://ruflo.ai) semantic memory — future research on related topics builds on prior findings automatically.

---

## Installation

**Requirements:** Python 3.9+, an Anthropic API key.

```bash
# 1. Clone
git clone https://github.com/MA1319Git/ora.git ~/tools/ora
cd ~/tools/ora

# 2. Create venv and install
python3 -m venv venv
venv/bin/python3 -m ensurepip --upgrade
venv/bin/pip install -e "."

# 3. Add the CLI wrapper to ~/.local/bin
mkdir -p ~/.local/bin
cat > ~/.local/bin/ora << 'EOF'
#!/bin/bash
exec "$HOME/tools/ora/venv/bin/python3" "$HOME/tools/ora/ora/engine.py" "$@"
EOF
chmod +x ~/.local/bin/ora

# 4. Add to PATH and PYTHONPATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/tools/ora:$PYTHONPATH"
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Enable live web search (optional)

Analysts can fetch real web sources before writing. Install the optional dep and add a [Tavily](https://app.tavily.com) API key (free tier: 1,000 searches/month):

```bash
venv/bin/pip install -e ".[web]"
export TAVILY_API_KEY=tvly-...
```

When `TAVILY_API_KEY` is set, each analyst automatically searches for sources on their angle and cites them inline. If the key is missing or `tavily-python` isn't installed, Ora falls back to knowledge-only mode silently.

---

## Usage

### CLI

```bash
ora "What are the systemic risks of AI in financial markets?"
ora --depth shallow "How does CRISPR base editing work?"
ora --no-save "What caused the 2008 financial crisis?"
ora --no-web "What is the philosophy of Stoicism?"
```

| Flag | Default | Description |
|---|---|---|
| `--depth shallow\|deep` | `deep` | `shallow` = faster (2k tokens/analyst), `deep` = thorough (6k tokens/analyst) |
| `--no-save` | off | Skip saving the report to `./reports/` |
| `--no-web` | off | Disable live web search; use model knowledge only |

Reports are saved to `./reports/YYYYMMDD_HHMMSS_<slug>.md` in your current directory.

### Python module

```python
from ora import research

report = research("What is the long-term impact of near-zero interest rates?")
print(report)
```

```python
# With options
report = research(
    question="How does transformer attention work?",
    depth="shallow",
    save=False,
    web=False,   # disable web search for this run
)
```

```python
# Point reports at a specific directory
from pathlib import Path

report = research(
    question="What are the key risks in mRNA drug delivery?",
    output_dir=Path("/my/project/research"),
)
```

### From another orchestrator

```python
from ora import research

# Use as a research tool inside a larger agentic pipeline
findings = research("What market signals preceded the 2020 crash?", depth="shallow", save=False)
# pass `findings` to your next agent
```

---

## Output format

Every report follows this structure:

```markdown
# [Descriptive Title]

## Executive Summary
3–5 sentences: core answer and key tensions.

## Key Findings
One subsection per major theme, integrating across all research angles.

## Uncertainties & Open Questions
What remains contested, unknown, or context-dependent.

## Conclusion
Direct answer to the original question + key implications.
```

---

## Project structure

```
~/tools/ora/
├── ora/
│   ├── __init__.py       # exports `research`
│   └── engine.py         # all agent logic
├── evals/                # evaluation harness
├── pyproject.toml
├── setup.py
└── venv/                 # local Python environment (not committed)

~/.local/bin/ora          # CLI wrapper script
```

---

## Why Ora is different

Most AI research is a single model answering in one shot. Ora is adversarial by design:

- **Specialization** — each agent has one job; a single prompt can't hold Scout, Analyst, and Critic simultaneously
- **Live web sources** — analysts fetch real pages before writing, so findings are grounded in current information with cited URLs
- **The Critic creates real tension** — a separate agent explicitly tasked with tearing apart the analysts' work
- **The Synthesizer resolves, not just reports** — it weighs the critique directly, correcting findings where challenges hold and pushing back where they don't
- **Memory compounds** — every session is stored semantically; related future queries build on prior work
- **Thinking enabled on hard agents** — Analysts and Synthesizer use Opus 4.7 with adaptive thinking

---

## Requirements

- Python 3.9+
- `anthropic >= 0.92.0`
- `ANTHROPIC_API_KEY` environment variable
- Optional: `tavily-python >= 0.3.0` + `TAVILY_API_KEY` for live web search (`pip install -e ".[web]"`)
- Optional: [Ruflo / claude-flow CLI](https://ruflo.ai) for persistent memory across sessions

---

## License

MIT
