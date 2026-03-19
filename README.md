<h1><div align="center">
 <img alt="pipecat subagents" width="500px" height="auto" src="https://github.com/pipecat-ai/pipecat-subagents/raw/refs/heads/main/pipecat-subagents.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-subagents)](https://pypi.org/project/pipecat-ai-subagents) ![Tests](https://github.com/pipecat-ai/pipecat-subagents/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat-subagents/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat-subagents) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/guides/features/pipecat-subagents) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Subagents is a distributed multi-agent framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme). Each agent runs its own Pipecat pipeline and communicates with other agents through a shared message bus, enabling you to decompose complex systems into specialized, coordinating agents that can run locally or across machines.

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
- **Any agent can own a transport.** A child agent might stream images or video through its own transport while the main agent handles voice.
- **A voice agent isn't required.** Agents can coordinate purely through tasks and bus messages for non-interactive pipelines.

## Key Concepts

### Bus

Agents communicate through a shared bus using pub/sub messaging. Place a `BusBridgeProcessor` in a pipeline to exchange frames with other agents across the bus.

| Class           | Description                                                                                                             |
|-----------------|-------------------------------------------------------------------------------------------------------------------------|
| `AgentBus`      | Abstract base for inter-agent messaging.                                                                                |
| `AsyncQueueBus` | In-process bus backed by `asyncio.Queue`s. No serialization overhead. This is the default bus created by `AgentRunner`. |
| `RedisBus`      | Distributed bus backed by Redis pub/sub for cross-process communication.                                                |

#### Serialization

Network buses need to serialize messages to bytes. Types that aren't JSON-native (e.g. `LLMContext`, `ToolsSchema`) require a `TypeAdapter` to convert them to/from JSON. Common adapters are registered by default.

| Class                   | Description                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| `JSONMessageSerializer` | Default serializer for network buses. Encodes messages as JSON.                                     |
| `TypeAdapter`           | Abstract base for serializing a specific type. Register via `MessageSerializer.register_adapter()`. |

#### Bridge

The `BusBridgeProcessor` is a Pipecat pipeline processor placed in an agent's pipeline (typically a transport/session agent) where an LLM would normally go. It sends non-lifecycle frames to the bus and injects incoming bus frames at its position, connecting the pipeline to whichever agent is active on the bus.

| Class                | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `BusBridgeProcessor` | Mid-pipeline processor that bridges frames between a pipeline and the bus. |

### Runner

The runner orchestrates the system: it creates pipeline tasks, manages agent lifecycle, and coordinates shutdown. Agents can be added dynamically at runtime.

| Class         | Description                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------------|
| `AgentRunner` | Entry point for running a multi-agent system. Owns the bus (or accepts one) and the agent registry. |

### Registry

The registry tracks which agents are ready. When a **root agent** (added via `AgentRunner.add_agent()`) becomes ready, the runner announces it to all other local agents via `on_agent_ready()`. In distributed setups, root agents are also announced to remote runners over the network bus.

**Child agents** (added via `BaseAgent.add_agent()`) are not broadcast. Only the parent is notified when a child is ready. Other agents can opt in via `watch_agent(name)`.

| Class           | Description                                                                             |
|-----------------|-----------------------------------------------------------------------------------------|
| `AgentRegistry` | Tracks ready agents (local and remote). Owned by the runner and shared with its agents. |

### Agents

Agents are the building blocks of a multi-agent system. Each agent connects to the bus and typically runs a Pipecat pipeline, exchanging frames and messages with other agents. Agents can also operate without a pipeline for lightweight coordination (e.g. task routing, agent factories). Agents can launch subagents, activate or deactivate each other, and coordinate work through tasks.

| Class                | Use when                                                                                                                                                                                |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `BaseAgent`          | You need a pipeline on the bus with no extra wiring. Handles lifecycle, parent-child, and task coordination.                                                                            |
| `DetachedAgent`      | Your pipeline receives frames from a `BusBridgeProcessor` in another agent. Adds bus frame routing and active/inactive state. Optionally use `handoff_to()` to transfer between agents. |
| `LLMDetachedAgent`   | Your detached agent needs an LLM. Adds `build_llm()`, `@tool` registration, and message injection on activation.                                                                        |
| `FlowsDetachedAgent` | Your detached agent needs structured conversation flows via [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows).                                                               |

#### Lifecycle hooks

| Hook | When it fires |
|---|---|
| `on_started()` | Agent is ready. Add child agents here. |
| `on_activated(args)` | Agent is activated via `activate_agent()`. |
| `on_deactivated()` | Agent is deactivated via `deactivate_agent()`. |
| `on_agent_ready()` | Another agent is ready to receive messages. |

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
