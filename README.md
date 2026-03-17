<h1><div align="center">
 <img alt="pipecat subagents" width="500px" height="auto" src="https://github.com/pipecat-ai/pipecat-subagents/raw/refs/heads/main/pipecat-subagents.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-subagents)](https://pypi.org/project/pipecat-ai-subagents) ![Tests](https://github.com/pipecat-ai/pipecat-subagents/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat-subagents/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat-subagents) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/guides/features/pipecat-subagents) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Subgents is a distributed multi-agent framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme). Each agent runs its own Pipecat pipeline and communicates with other agents through a shared message bus, enabling you to decompose complex systems into specialized, coordinating agents that can run locally or across machines.

Whether local or distributed, the programming model is the same: create an `AgentRunner`, connect it to the bus, and add agents.

## Installation

```bash
uv add pipecat-ai-agents

# or: pip install pipecat-ai-agents
```

> Requires Python 3.10+ and [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started).

## Architecture

Agents communicate through a shared **AgentBus**. The diagram below shows a common voice-first topology:

```
        ┌───────────────────┐    ┌───────────────────┐    ┌──────────────────┐
        │    Main Agent     │    │   Voice Agent     │    │   Worker(s)      │
        │                   │    │                   │    │                  │
        │  Transport I/O    │    │   LLM + @tool     │    │  Task Handlers   │
        │  STT → BusBridge  │    │                   │    │                  │
        │  BusBridge → TTS  │    │                   │    │                  │
        └─────────┬─────────┘    └─────────┬─────────┘    └────────┬─────────┘
              messages                 messages                 messages
                  │                        │                       │
        ══════════╧════════════════════════╧═══════════════════════╧══════════
                                      Agent Bus
        ══════════════════════════════════════════════════════════════════════
```

- **Main agent** owns the transport (audio I/O) and places a `BusBridgeProcessor` where an LLM would normally go. Messages flow through the bus to whichever child agent is active.
- **Voice agent** runs an LLM with tools. It handles conversation and dispatches work to other agents.
- **Worker agents** receive tasks, process them, and return results. A voice agent can spawn multiple workers in parallel.
- **Any agent can own a transport** — a child agent might stream images or video through its own transport while the main agent handles voice.
- **A voice agent isn't required** — agents can coordinate purely through tasks and bus messages for non-interactive pipelines.

## Key Concepts

### Agents

Each agent runs its own Pipecat pipeline and communicates via the bus.

- **`BaseAgent`** — Abstract base. Handles bus subscription, pipeline lifecycle, parent-child relationships, activation, and task coordination.
- **`LLMAgent`** — Agent with an LLM pipeline. Register tools with `@tool`, inject messages on activation, and transfer between agents.
- **`FlowsAgent`** — Agent that integrates [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for structured, node-based conversations.

#### Lifecycle hooks

| Hook                       | When it fires                                            |
|----------------------------|----------------------------------------------------------|
| `on_agent_started()`       | Pipeline is ready. Add child agents here.                |
| `on_agent_registered()`    | A child agent is on the bus and ready. Activate it here. |
| `on_agent_activated(args)` | Agent is activated (receives frames).                    |
| `on_agent_deactivated()`   | Agent is deactivated (stops receiving frames).           |

### Bus

Pub/sub communication between agents and the runner.

- **`AgentBus`** — Abstract base for inter-agent messaging.
- **`BusBridgeProcessor`** — Mid-pipeline processor that bridges frames across agent boundaries. Non-lifecycle frames go to the bus; bus frames are injected at the bridge position.

#### Local buses

In-process buses for agents running in the same Python process.

- **`AsyncQueueBus`** — Fan-out bus backed by per-subscriber `asyncio.Queue`s. No serialization overhead — messages are passed as Python objects. This is the default bus created by `AgentRunner`.

#### Network buses

Distributed buses for agents running across separate processes or machines. Network buses require a `MessageSerializer` to convert messages to/from bytes. Frame payloads inside `BusFrameMessage` are handled by pluggable `FrameAdapter`s registered on the serializer.

- **`RedisBus`** — Bus backed by Redis pub/sub. Each subscriber gets its own Redis subscription. Messages marked with `BusLocalMixin` (e.g. `BusAddAgentMessage`) are silently skipped since they carry in-memory references.
- **`JSONMessageSerializer`** — Default serializer that encodes messages as JSON. Register a `FrameAdapter` per frame type to handle serialization of Pipecat frames.

### Tasks

Tasks let a parent agent spawn workers, wait for results, and optionally cancel or time out. Workers can send a single response or stream incremental results.

```
                        Parent                              Worker
                           │                                  │
                           │ start_task(payload, timeout)     │
                           ├─────────────────────────────────►│ on_task_request()
                           │                                  │
                           │ request_task_update()            │
                           ├─────────────────────────────────►│ on_task_update_requested()
                           │               send_task_update() │ (optional)
          on_task_update() │◄─────────────────────────────────┤
                           │                                  │
                           │               ...                │
                           │                                  │
                           │               send_task_update() │ (optional)
          on_task_update() │◄─────────────────────────────────┤
                           │                                  │
                           │         send_task_stream_start() │
    on_task_stream_start() │◄─────────────────────────────────┤ ┐
                           │          send_task_stream_data() │ │ streaming
     on_task_stream_data() │◄─────────────────────────────────┤ │ (optional)
                           │           send_task_stream_end() │ │
      on_task_stream_end() │◄─────────────────────────────────┤ ┘
                           │                                  │
                           │       send_task_response(status) │
        on_task_response() │◄─────────────────────────────────┤
       on_task_completed() │                                  │
```

#### Task hooks

| Hook                              | When it fires                                                                     |
|-----------------------------------|-----------------------------------------------------------------------------------|
| `on_task_request()`               | Worker: received work from a parent.                                              |
| `on_task_update_requested()`      | Worker: parent requested a progress update. Respond with `send_task_update()`.    |
| `on_task_cancelled()`             | Worker: parent cancelled this task. A `CANCELLED` response is sent automatically. |
| `on_task_response()`              | Parent: a worker sent a response.                                                 |
| `on_task_completed()`             | Parent: all workers in the task group have responded.                             |
| `on_task_stream_start/data/end()` | Parent: a worker is streaming incremental results.                                |

### Runner

- **`AgentRunner`** — Orchestrates agent lifecycle, creates pipeline tasks, and coordinates shutdown. Agents can be added dynamically at runtime.

## Examples

The [examples](examples/) directory includes complete working implementations. See the [examples README](examples/README.md) for setup and running instructions.

## Contributing to the framework

1. Clone the repository and navigate to it:

   ```bash
   git clone https://github.com/pipecat-ai/pipecat-agents.git
   cd pipecat-agents
   ```

2. Install development dependencies:

   ```bash
   uv sync --group dev
   ```

3. Install the git pre-commit hooks (these help ensure your code follows project rules):

   ```bash
   uv run pre-commit install
   ```

   > The package is automatically installed in editable mode when you run `uv sync`.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat-agents/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Want to contribute code?** Check our [CONTRIBUTING.md](CONTRIBUTING.md) guide
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Reach us on X](https://x.com/pipecat_ai)
