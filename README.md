<h1><div align="center">
 <img alt="pipecat subagents" width="500px" height="auto" src="https://github.com/pipecat-ai/pipecat-subagents/raw/refs/heads/main/pipecat-subagents.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-subagents)](https://pypi.org/project/pipecat-ai-subagents) ![Tests](https://github.com/pipecat-ai/pipecat-subagents/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat-subagents/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat-subagents) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/guides/features/pipecat-subagents) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Subagents is a distributed multi-agent framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme). Each agent runs its own Pipecat pipeline and communicates with other agents through a shared message bus, enabling you to decompose complex systems into specialized, coordinating agents that can run locally or across machines.

Whether local or distributed, the programming model is the same: create an `AgentRunner`, connect it to the bus, and add agents.

## Features

| Feature                   | Description                                                                                           |
|---------------------------|-------------------------------------------------------------------------------------------------------|
| **Run anywhere**          | In-process, multi-process, across machines, or externally via proxies                                 |
| **Bus communication**     | Agents communicate via a shared bus (local or through network)                                        |
| **Pipecat pipelines**     | Each agent runs its own Pipecat pipeline                                                              |
| **Tasks and task groups** | Parallel workers with streaming, timeouts, and auto-cancel on error                                   |
| **Framework integration** | Wrap other agent frameworks as processors or subagents                                                |
| **Agent handoff**         | Transfer control between agents mid-conversation                                                      |
| **Structured flows**      | [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) integration for node-based conversations |
| **Error handling**        | Pipeline errors propagate to parents with configurable recovery                                       |

## Installation

```bash
uv add pipecat-ai-subagents

# or: pip install pipecat-ai-subagents
```

> Requires Python 3.10+ and [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started).

## Examples

See the [examples](examples/) directory for complete, runnable demos.

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

#### Messages

Bus messages are Pipecat frames, which means they carry the same priority semantics as pipeline frames. Messages extend either `BusDataMessage` (normal priority, based on `DataFrame`) or `BusSystemMessage` (high priority, based on `SystemFrame`). Each subscriber has its own priority queue, so system messages like cancellations are delivered before queued data messages.

#### Serialization

Network buses need to serialize messages to bytes. Types that aren't JSON-native (e.g. `LLMContext`, `ToolsSchema`) require a `TypeAdapter` to convert them to/from JSON. Common adapters are registered by default.

| Class                   | Description                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| `JSONMessageSerializer` | Default serializer for network buses. Encodes messages as JSON.                                     |
| `TypeAdapter`           | Abstract base for serializing a specific type. Register via `MessageSerializer.register_adapter()`. |

#### Bridge

The `BusBridgeProcessor` is a Pipecat pipeline processor placed in an agent's pipeline (typically a transport/session agent) where an LLM would normally go. It sends non-lifecycle frames to the bus and injects incoming bus frames at its position, connecting the pipeline to whichever agent is active on the bus. Child agents that participate in this frame flow use `bridged=()`, which adds edge processors that route frames to and from the bus.

Bridges support named routing for flexible topologies. Give each bridge a name (e.g. `BusBridgeProcessor(bridge="voice")`) and child agents specify which bridges they accept frames from via `bridged=("voice",)`. This enables parallel pipelines with independent frame streams, or multiple agents processing frames from the same bridge.

| Class                | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `BusBridgeProcessor` | Mid-pipeline processor that bridges frames between a pipeline and the bus. |

### Runner

The runner orchestrates the system: it creates pipeline tasks, manages agent lifecycle, and coordinates shutdown. Agents can be added dynamically at runtime.

| Class         | Description                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------------|
| `AgentRunner` | Entry point for running a multi-agent system. Owns the bus (or accepts one) and the agent registry. |

### Registry and visibility

Only **root agents** (added via `AgentRunner.add_agent()`) are visible across the system. When a root agent becomes ready, the runner announces it to all local agents and to remote runners over the network bus.

**Child agents** (added via `BaseAgent.add_agent()`) are private to their parent. Only the parent is notified when a child is ready via `on_agent_ready()`.

Use `watch_agent(name)` to request notification when a specific agent registers.

| Class           | Description                                                                             |
|-----------------|-----------------------------------------------------------------------------------------|
| `AgentRegistry` | Tracks ready agents (local and remote). Owned by the runner and shared with its agents. |

### Agents

Agents are the building blocks of a multi-agent system. Each agent connects to the bus and runs a Pipecat pipeline, exchanging frames and messages with other agents. Agents can launch subagents, activate or deactivate each other, and coordinate work through tasks.

| Class        | Use when                                                                                                                                                                                |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `BaseAgent`  | You need a pipeline on the bus with no extra wiring. Handles lifecycle, parent-child, and task coordination. Pass `bridged=()` to add edge processors for bus frame routing.          |
| `LLMAgent`   | Your agent needs an LLM. Adds `build_llm()`, `@tool` registration, and message injection on activation. Pass `bridged=()` for agents that receive frames from a `BusBridgeProcessor`. |
| `FlowsAgent` | Your agent needs structured conversation flows via [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows). Always bridged.                                                        |

#### Naming

Every agent has a unique name passed at construction. Names are used for bus message targeting, activation, task routing, and logging. Choose short, descriptive names (e.g. `"greeter"`, `"support"`, `"worker"`). In distributed setups, agent names must be unique across all runners.

#### Agent lifecycle

Hooks about this agent's own state.

| Hook                     | When it fires                                           |
|--------------------------|---------------------------------------------------------|
| `on_ready()`             | Agent is ready to operate.                              |
| `on_finished()`          | Agent's pipeline has finished.                          |
| `on_error(error, fatal)` | A pipeline error occurred.                              |
| `on_activated(args)`     | Agent is activated via `activate_agent()`.              |
| `on_deactivated()`       | Agent is deactivated via `deactivate_agent()`.          |

#### Other agent events

Hooks about other agents in the system.

| Hook                         | When it fires                                       |
|------------------------------|-----------------------------------------------------|
| `on_agent_ready(ready_info)` | Another agent is ready to receive messages.         |
| `on_agent_error(error_info)` | A child agent reported an error via `send_error()`. |

#### Error handling

Errors are not propagated automatically. When a pipeline error occurs, `on_error(error, fatal)` fires on the agent. The agent decides how to respond: recover, fail a running task via `send_task_response(status=TaskStatus.ERROR)`, or escalate to the parent via `send_error()`.

`send_error()` follows the same visibility rules as readiness: child agent errors stay local (never cross the network), root agent errors are broadcast. The parent receives `on_agent_error(error_info)` for child errors.

For task groups, `cancel_on_error=True` (the default) automatically cancels all workers in the group if any worker responds with `ERROR` or `FAILED` status.

### Tasks

Tasks let a parent agent dispatch work to one or more workers, wait for results, and optionally cancel or time out. Workers can send a single response or stream incremental results.

A parent sends work with `request_task()` (fire-and-forget) or `task()` (structured context manager that waits for the response). Workers receive `on_task_request()`, do work, and reply via `send_task_response()`. For dispatching to multiple workers in parallel, see [Task groups](#task-groups).

```
                        Parent                              Worker
                           │                                  │
                           │ request_task(payload, timeout)   │
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

#### Task handlers

The `@task` decorator provides declarative routing and parallel execution for task requests. Named handlers receive only matching requests; unnamed handlers are the default fallback. Use `parallel=True` to run each request in its own asyncio task.

#### Task hooks

| Hook                              | When it fires                                                                     |
|-----------------------------------|-----------------------------------------------------------------------------------|
| `on_task_request()`               | Worker: received work from a parent.                                              |
| `on_task_update_requested()`      | Worker: parent requested a progress update. Respond with `send_task_update()`.    |
| `on_task_cancelled()`             | Worker: parent cancelled this task. A `CANCELLED` response is sent automatically. |
| `on_task_response()`              | Parent: a worker sent a response.                                                 |
| `on_task_completed()`             | Parent: all workers in the task group have responded.                             |
| `on_task_stream_start/data/end()` | Parent: a worker is streaming incremental results.                                |

### Task groups

A task group dispatches work to multiple workers in parallel. `task_group()` returns a context manager that waits for all workers to respond. It supports timeouts, automatic cancellation on worker error, and an async iterator for receiving intermediate events (updates and streaming data) while waiting for completion. Results are collected in `responses` keyed by agent name. `request_task_group()` is also available for fire-and-forget dispatch.

### Proxy Agents

Proxy agents connect two independent bus instances that can't communicate directly. Each proxy is scoped to a specific agent name, so only messages for that agent cross the connection. This enables agents to operate across separate networks, third-party servers, or isolated processes without sharing a bus.

```
    +-------------+    +-------------+           +-------------+     +--------------+
    |             |    |             |           |             |     |              |
    | Local Agent |    | Proxy Agent |  <~~~~~>  | Proxy Agent |     | Remote Agent |
    |             |    |             |           |             |     |              |
    +-------------+    +-------------+           +-------------+     +--------------+
        messages           messages                  messages            messages
            │                 │                         │                   │
  ══════════╧═════════════════╧════════         ════════╧═══════════════════╧═════════
                Agent Bus                                     Agent Bus
  ═════════════════════════════════════         ══════════════════════════════════════
```

The framework includes a WebSocket proxy implementation. Other transports can be added by following the same pattern.

| Class                       | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `WebSocketProxyClientAgent` | Connects to a remote server and forwards bus messages for a specific agent. |
| `WebSocketProxyServerAgent` | Accepts a WebSocket connection and routes messages to/from a local agent.   |

Agent discovery works automatically: when the remote agent is ready, the proxy notifies the local side.

#### Security

Each proxy filters messages by agent name. Only the following cross the connection:

- **Targeted messages** between the two configured agents (activation, tasks, errors).
- **Registry messages** for agent discovery (sent automatically when the remote agent is ready).
- **Additional message types** opted in via `forward_messages` (e.g. `BusFrameMessage` for frame routing). These are forwarded based on source agent name only, regardless of target.

By default, only targeted messages and registry messages cross the connection. Frame routing must be explicitly enabled by passing `forward_messages` to the proxy constructor. Everything else is blocked: local-only messages, broadcast lifecycle messages (end/cancel), and messages for other agents. Closing the connection signals shutdown.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat-subagents/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Reach us on X](https://x.com/pipecat_ai)
