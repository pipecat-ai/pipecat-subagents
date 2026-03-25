# Parallel Debate (Local)

Demonstrates parallel task workers using a task group context. A voice agent receives a topic from the user and spawns three worker agents in parallel.  Each worker runs its own LLM pipeline with a persistent context, generating a perspective: one argues in favor, one against, and one provides neutral analysis. The voice agent collects all three and synthesizes a balanced response.

## Architecture

```
Debate Agent (transport + BusBridge)
  └── Moderator Agent (LLM, bridged)
        └── @tool debate(topic)
              └── request_task_group(advocate, critic, analyst)
                     └── Debate Worker (LLM + context aggregators)
```

- **Debate Agent**: Owns the transport (STT, TTS) and bridges frames to the bus.
- **Moderator Agent**: Bridged LLM agent that talks to the user. Has a `debate` tool that spawns parallel workers.
- **Debate Workers**: Each runs its own LLM pipeline with a persistent context. Workers remember previous topics across multiple debate rounds.

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
uv run examples/local/parallel-debate/parallel_debate.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

To use Daily transport:

```bash
uv run examples/local/parallel-debate/parallel_debate.py --transport daily
```

## Environment variables

| Variable           | Required by                  |
|--------------------|------------------------------|
| `OPENAI_API_KEY`   | All agents                   |
| `DEEPGRAM_API_KEY` | STT                          |
| `CARTESIA_API_KEY` | TTS                          |
| `DAILY_API_KEY`    | Only with `--transport daily`|
