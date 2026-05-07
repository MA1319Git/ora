# Ora — Multi-Agent Deep Research Engine

Ora turns a single question into a comprehensive research report by running five specialized AI agents in sequence. Each agent has a distinct role, a tuned system prompt, and the right model for its job — the result is a report that has been challenged and defended before you read it.

```
Scout → Analyst Swarm → Critic → Consensus → Synthesizer
```

---

## How it works

| Agent | Model | Job |
|---|---|---|
| **Scout** | Sonnet 4.6 | Breaks the question into 3–5 orthogonal research angles |
| **Analyst Swarm** | Opus 4.7 + thinking | Each analyst dives deep on one angle independently |
| **Critic** | Sonnet 4.6 | Stress-tests all findings — gaps, contradictions, unsupported claims |
| **Consensus** | Opus 4.7 + thinking | Analysts respond to each challenge and refine findings |
| **Synthesizer** | Opus 4.7 + thinking | Merges everything into a structured markdown report |

Every session is stored in [Ruflo](https://ruflo.ai) semantic memory — future research on related topics builds on prior findings automatically.

---

## Installation

**Requirements:** Python 3.9+, an Anthropic API key.

```bash
# 1. Clone
git clone https://github.com/MA1319Git/ora.git ~/tools/ora
cd ~/tools/ora

# 2. Create venv and install dependencies
python3 -m venv venv
venv/bin/python3 -m ensurepip --upgrade
venv/bin/pip install anthropic

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

---

## Usage

### CLI

```bash
ora "What are the systemic risks of AI in financial markets?"
ora --depth shallow "How does CRISPR base editing work?"
ora --no-save "What caused the 2008 financial crisis?"
```

| Flag | Default | Description |
|---|---|---|
| `--depth shallow\|deep` | `deep` | `shallow` = faster (2k tokens/analyst), `deep` = thorough (6k tokens/analyst) |
| `--no-save` | off | Skip saving the report to `./reports/` |

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
├── pyproject.toml
├── setup.py
└── venv/                 # local Python environment (not committed)

~/.local/bin/ora          # CLI wrapper script
```

---

## Why Ora is different

Most AI research is a single model answering in one shot. Ora is adversarial by design:

- **Specialization** — each agent has one job; a single prompt can't hold Scout, Analyst, and Critic simultaneously
- **The Critic creates real tension** — a separate agent explicitly tasked with tearing apart the analysts' work
- **Consensus forces refinement** — analysts respond to every challenge before the Synthesizer writes the final report
- **Memory compounds** — every session is stored semantically; related future queries build on prior work
- **Thinking enabled on hard agents** — Analysts, Rebuttal, and Synthesizer use Opus 4.7 with adaptive thinking

---

## Requirements

- Python 3.9+
- `anthropic >= 0.92.0`
- `ANTHROPIC_API_KEY` environment variable
- Optional: [Ruflo / claude-flow CLI](https://ruflo.ai) for persistent memory across sessions

---

## License

MIT
