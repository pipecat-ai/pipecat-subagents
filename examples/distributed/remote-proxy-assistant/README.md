# Distributed LLM Agent via WebSocket Proxy

Runs an LLM agent on a remote server, connected to the main transport agent via a WebSocket proxy. No shared bus (Redis) required, the proxy handles message forwarding over a point-to-point WebSocket connection.

## Architecture

```
      +---------------+      +-------------------+           +-------------------+      +-----------------+
      |               |      |                   |           |                   |      |                 |
      |  Main Agent   |      |    Proxy Agent    |  <~~~~~>  |    Proxy Agent    |      | Assistant Agent |
      |               |      |                   |           |                   |      |                 |
      +---------------+      +-------------------+           +-------------------+      +-----------------+
          messages                 messages                        messages                   messages
              │                       │                               │                          │
    ══════════╧═══════════════════════╧════════════         ══════════╧══════════════════════════╧══════════
                       Agent Bus                                                Agent Bus
    ═══════════════════════════════════════════════         ════════════════════════════════════════════════
```

- **main_agent.py**: Transport agent with STT, TTS, and a BusBridge. Creates a `WebSocketProxyClientAgent` that connects to the remote server.
- **assistant_agent.py**: FastAPI server. Each WebSocket connection creates a `WebSocketProxyServerAgent` and a bridged `AcmeAssistant` LLM agent.

## Quick start (single machine)

```bash
# Terminal 1: start the remote assistant agent
uv run assistant_agent.py

# Terminal 2: start the main transport agent
uv run main_agent.py --remote-agent-url ws://localhost:8765/ws
```

Open http://localhost:7860/client in your browser to talk to the bot.

## Running across machines

```bash
# Server machine: start the assistant agent
uv run assistant_agent.py --host 0.0.0.0 --port 8765

# Client machine: point at the server
uv run main_agent.py --remote-agent-url ws://server-host:8765/ws
```

## Security

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

## Environment variables

| Variable           | Required by        |
|--------------------|--------------------|
| `OPENAI_API_KEY`   | assistant_agent.py |
| `DEEPGRAM_API_KEY` | main_agent.py      |
| `CARTESIA_API_KEY` | main_agent.py      |
