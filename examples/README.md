# Pipecat Subagents Examples

This directory contains example implementations demonstrating the Pipecat Subagents framework.

# Setup

From the main repo directory:

```bash
uv sync --all-extras
source .venv/bin/activate.sh

cd examples
```

Copy `env.example` to `.env` and add your API keys:

```bash
cp env.example .env
```

# Environment variables

| Variable           | Required by                             |
|--------------------|-----------------------------------------|
| `OPENAI_API_KEY`   | Voice Agent (LLM)                       |
| `DEEPGRAM_API_KEY` | STT                                     |
| `CARTESIA_API_KEY` | TTS                                     |
| `DAILY_API_KEY`    | Optional: Only with `--transport daily` |

# Table of Contents

**[Local](#local)** (single process)
- [Agent Handoff](#agent-handoff)
- [Parallel Debate](#parallel-debate)
- [Voice Code Assistant with Claude Agent SDK](#voice-code-assistant)

**[Distributed](#distributed)** (multi-process)
- [Agent Handoff via Redis](#agent-handoff-via-redis)
- [LLM Agent via WebSocket Proxy](#llm-agent-via-websocket-proxy)

# Local

Examples where all agents run in the same process.

## Agent Handoff

Demonstrates agents that transfer control between each other during a voice conversation. All agents run in the same process on a local bus.

### Running

```bash
uv run local/agent-handoff/two_llm_agents.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

To use Daily transport:

```bash
uv run local/agent-handoff/two_llm_agents.py --transport daily
```

### Overview

- **[`single_agent.py`](local/agent-handoff/single_agent.py)**: Simplest usage: a single agent running a complete voice pipeline (transport, STT, LLM, TTS) through the AgentRunner. No multi-agent coordination.
- **[`two_llm_agents.py`](local/agent-handoff/two_llm_agents.py)**: Two LLM agents (greeter + support) that transfer control between each other. A main agent bridges transport I/O to the bus.
- **[`two_llm_agents_with_tts.py`](local/agent-handoff/two_llm_agents_with_tts.py)**: Same as above, but each LLM agent has its own TTS with a distinct voice. The main agent has no TTS.
- **[`llm_and_flows_agent.py`](local/agent-handoff/llm_and_flows_agent.py)**: Mixing agent types: an LLM agent (router) and a Flows agent (restaurant reservation with structured nodes). Agents transfer between each other.

## Parallel Debate

Demonstrates parallel task workers using a task group context. A voice agent receives a topic from the user and spawns three worker agents in parallel.  Each worker runs its own LLM pipeline with a persistent context, generating a perspective: one argues in favor, one against, and one provides neutral analysis. The voice agent collects all three and synthesizes a balanced response.

### Running

```bash
uv run local/parallel-debate/parallel_debate.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

To use Daily transport:

```bash
uv run local/parallel-debate/parallel_debate.py --transport daily
```

### Architecture

```
Debate Agent (transport + BusBridge)
  └── Moderator Agent (LLM, bridged)
        └── @tool debate(topic)
              └── task_group(advocate, critic, analyst)
                     └── Debate Worker (LLM + context aggregators)
```

- **Debate Agent**: Owns the transport (STT, TTS) and bridges frames to the bus.
- **Moderator Agent**: Bridged LLM agent that talks to the user. Has a `debate` tool that spawns parallel workers.
- **Debate Workers**: Each runs its own LLM pipeline with a persistent context. Workers remember previous topics across multiple debate rounds.

## Voice Code Assistant

Talk to your codebase hands-free. Ask questions about code, project structure, or file contents and get spoken answers based on actual files. The Claude Agent SDK worker navigates the filesystem using Read, Write, Bash, Glob, and Grep tools.

### Additional environment variables

| Variable            | Required by                    |
|---------------------|--------------------------------|
| `ANTHROPIC_API_KEY` | Code Worker (Claude Agent SDK) |
| `PROJECT_PATH`      | Optional, defaults to cwd      |

### Running

```bash
# Default: explores the current directory
uv run local/code-assistant/code_assistant.py

# Specify a project path
PROJECT_PATH=/path/to/your/project uv run local/code-assistant/code_assistant.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

To use Daily transport:

```bash
uv run local/code-assistant/code_assistant.py --transport daily
```

### Example questions

- "What does the main module do?"
- "Find all TODO comments in the project"
- "How is error handling implemented?"
- "What dependencies does this project use?"
- "Explain the test structure"

### Architecture

```
CodeAssistant (transport + BusBridge)
  └── Voice Agent (LLM, bridged)
        └── @tool ask_code(question)
              └── Code Worker (Claude Agent SDK)
```

- **CodeAssistant** ([`code_assistant.py`](local/code-assistant/code_assistant.py)): Owns the transport (STT, TTS) and bridges frames to the bus.
- **VoiceAgent** ([`voice_agent.py`](local/code-assistant/voice_agent.py)): Bridged LLM agent that talks to the user. Has an `ask_code` tool that dispatches questions to the code worker using `request_task()`. The agent stays conversational while the worker runs in the background.
- **CodeWorker** ([`code_worker.py`](local/code-assistant/code_worker.py)): Runs a Claude Agent SDK session with filesystem tools to explore the project and answer questions.

# Distributed

Examples where agents run across separate processes or machines.

## Agent Handoff via Redis

Same two-agent handoff as `examples/agent-handoff/two_llm_agents.py`, but
each agent runs as a separate process connected via Redis pub/sub.

### Quick start (single machine, local Redis)

```bash
# Terminal 1: start Redis
docker run --rm -p 6379:6379 redis:7

# Terminal 2: start the greeter agent
uv run llm_agent.py greeter

# Terminal 3: start the support agent
uv run llm_agent.py support

# Terminal 4: start the main transport agent
uv run main_agent.py
```

All three agent processes connect to `redis://localhost:6379` by default.

### Running across machines

Point each process at the same Redis instance:

```bash
# Machine A
uv run main_agent.py --redis-url redis://your-redis-host:6379

# Machine B
uv run llm_agent.py greeter --redis-url redis://your-redis-host:6379

# Machine C
uv run llm_agent.py support --redis-url redis://your-redis-host:6379
```

### Architecture

```
Machine A                    Redis                  Machine B
+------------+          +-------------+          +-------------+
| main_agent |  <---->  | pub/sub     |  <---->  |  llm_agent  |
| (transport,|          | channel:    |          |  (greeter)  |
|  STT, TTS) |          | pipecat:acme|          +-------------+
+------------+          +-------------+          +-------------+
                               ^                 |  llm_agent  |
                               +-------------->  |  (support)  |
                                                 +-------------+
```

- **[main_agent.py](distributed/redis-handoff/main_agent.py)** — Transport agent: Daily WebRTC, Deepgram STT, Cartesia TTS
- **[llm_agent.py](distributed/redis-handoff/llm_agent.py)** — LLM agent: runs either `greeter` or `support` with OpenAI


## LLM Agent via WebSocket Proxy

Runs an LLM agent on a remote server, connected to the main transport agent via a WebSocket proxy. No shared bus (Redis) required, the proxy handles message forwarding over a point-to-point WebSocket connection.

### Quick start (single machine)

```bash
# Terminal 1: start the remote assistant agent
uv run assistant_agent.py

# Terminal 2: start the main transport agent
uv run main_agent.py --remote-agent-url ws://localhost:8765/ws
```

Open http://localhost:7860/client in your browser to talk to the bot.

### Running across machines

```bash
# Server machine: start the assistant agent
uv run assistant_agent.py --host 0.0.0.0 --port 8765

# Client machine: point at the server
uv run main_agent.py --remote-agent-url ws://server-host:8765/ws
```

### Architecture

```
    +-------------+    +-------------+           +-------------+     +-----------------+
    |             |    |             |           |             |     |                 |
    | Main Agent  |    | Proxy Agent |  <~~~~~>  | Proxy Agent |     | Assistant Agent |
    |             |    |             |           |             |     |                 |
    +-------------+    +-------------+           +-------------+     +-----------------+
        messages           messages                  messages             messages
            │                 │                         │                    │
  ══════════╧═════════════════╧════════         ════════╧════════════════════╧═══════════
                Agent Bus                                     Agent Bus
  ═════════════════════════════════════         ═════════════════════════════════════════
```

- **[main_agent.py](distributed/remote-proxy-assistant/main_agent.py)**: Transport agent with STT, TTS, and a BusBridge. Creates a `WebSocketProxyClientAgent` that connects to the remote server.
- **[assistant_agent.py](distributed/remote-proxy-assistant/assistant_agent.py)**: FastAPI server. Each WebSocket connection creates a `WebSocketProxyServerAgent` and a bridged `AcmeAssistant` LLM agent.

### Security

The proxy agents filter messages by agent name:
- Only messages targeted at the remote agent cross the WebSocket
- Only messages targeted at the local agent are accepted from the WebSocket
- Broadcast messages never cross the WebSocket

Pass HTTP headers for authentication:
```python
proxy = WebSocketProxyClientAgent(
    "proxy",
    bus=bus,
    url="wss://server-host:8765/ws",
    remote_agent_name="assistant",
    local_agent_name="acme",
    headers={"Authorization": "Bearer <token>"},
)
```
