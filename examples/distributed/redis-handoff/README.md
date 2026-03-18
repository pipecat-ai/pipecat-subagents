# Distributed Agent Handoff via Redis

Same two-agent handoff as `examples/agent-handoff/two_llm_agents.py`, but
each agent runs as a separate process connected via Redis pub/sub.

## Architecture

```
Machine A                    Redis                    Machine B
+-----------+          +-------------+          +-------------+
| main_agent|  <---->  | pub/sub     |  <---->  | llm_agent   |
| (transport,|          | channel:    |          | (greeter)   |
|  STT, TTS) |          | pipecat:acme|          +-------------+
+-----------+          +-------------+          +-------------+
                            ^                   | llm_agent   |
                            +-----------------> | (support)   |
                                                +-------------+
```

- **main_agent.py** — Transport agent: Daily WebRTC, Deepgram STT, Cartesia TTS
- **llm_agent.py** — LLM agent: runs either `greeter` or `support` with OpenAI

## Quick start (single machine, local Redis)

```bash
# Terminal 1: start Redis
docker run --rm -p 6379:6379 redis:7

# Terminal 2: start the greeter agent
python llm_agent.py greeter

# Terminal 3: start the support agent
python llm_agent.py support

# Terminal 4: start the main transport agent
python main_agent.py
```

All three agent processes connect to `redis://localhost:6379` by default.

## Running across machines

Point each process at the same Redis instance:

```bash
# Machine A
python main_agent.py --redis-url redis://your-redis-host:6379

# Machine B
python llm_agent.py greeter --redis-url redis://your-redis-host:6379

# Machine C
python llm_agent.py support --redis-url redis://your-redis-host:6379
```

## Environment variables

| Variable | Required by |
|---|---|
| `OPENAI_API_KEY` | llm_agent.py |
| `DEEPGRAM_API_KEY` | main_agent.py |
| `CARTESIA_API_KEY` | main_agent.py |
| `DAILY_ROOM_URL` | main_agent.py |
| `DAILY_ROOM_TOKEN` | main_agent.py |
