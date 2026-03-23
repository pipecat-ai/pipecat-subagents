# Agent Handoff (Local)

Demonstrates agents that transfer control between each other during a
voice conversation. All agents run in the same process on a local bus.

## Examples

- **`single_agent.py`**: Simplest usage: a single agent running a complete voice pipeline (transport, STT, LLM, TTS) through the AgentRunner. No multi-agent coordination.
- **`two_llm_agents.py`**: Two LLM agents (greeter + support) that transfer control between each other. A main agent bridges transport I/O to the bus.
- **`two_llm_agents_with_tts.py`**: Same as above, but each LLM agent has its own TTS with a distinct voice. The main agent has no TTS.
- **`llm_and_flows_agent.py`**: Mixing agent types: an LLM agent (router) and a Flows agent (restaurant reservation with structured nodes). Agents transfer between each other.

## Setup

```bash
uv sync
uv pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero,examples]"
```

Copy `env.example` to `.env` and add your API keys:

```bash
cp examples/env.example .env
```

## Running

```bash
uv run examples/local/agent-handoff/single_agent.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

To use Daily transport:

```bash
uv run examples/local/agent-handoff/single_agent.py --transport daily
```

## Environment variables

| Variable           | Required by                  |
|--------------------|------------------------------|
| `OPENAI_API_KEY`   | All examples                 |
| `DEEPGRAM_API_KEY` | All examples                 |
| `CARTESIA_API_KEY` | All examples                 |
| `DAILY_API_KEY`    | Only with `--transport daily`|
