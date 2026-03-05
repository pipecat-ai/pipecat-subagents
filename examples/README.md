# Pipecat Agents Examples

This directory contains example implementations demonstrating the Pipecat Agents framework.

## Agent Handoff

The [`agent-handoff/`](agent-handoff/) examples demonstrate agents that transfer control between each other during a conversation.

- **`single_agent.py`** — Simplest usage: a single agent running a complete voice pipeline (transport, STT, LLM, TTS) through the AgentRunner. No multi-agent coordination.
- **`two_llm_agents.py`** — Two LLM agents (greeter + support) that transfer control between each other. A main agent bridges transport I/O to the bus.
- **`two_llm_agents_with_tts.py`** — Same as above, but each LLM agent has its own TTS with a distinct voice. The main agent has no TTS.
- **`llm_and_flows_agent.py`** — Mixing agent types: an LLM agent (router) and a Flows agent (restaurant reservation with structured nodes). Agents transfer between each other.

## Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### 1. Installation

```bash
uv sync
uv pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero,examples]"
```

### 2. Configuration

Copy `env.example` to `.env` and add your API keys:

```bash
cp env.example .env
```

Required keys:
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`
- `CARTESIA_API_KEY`
- `DAILY_API_KEY` (for Daily transport)

### 3. Running Examples

```bash
uv run examples/agent-handoff/single_agent.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

### Other Transports

Examples default to SmallWebRTC. To use Daily:

```bash
uv run examples/agent-handoff/single_agent.py --transport daily
```
