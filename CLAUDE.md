# Pipecat Subagents

Distributed multi-agent framework for [Pipecat](https://github.com/pipecat-ai/pipecat). Each agent runs its own Pipecat pipeline and communicates with other agents through a shared message bus.

## Commands

```bash
uv sync --group dev          # Install dependencies
uv run pytest                # Run tests
uv run ruff check .          # Lint
uv run ruff format           # Format
```

## Architecture

Agents communicate through a shared `AgentBus`. A typical voice-first system has:

- **Main agent** (`BaseAgent`): owns the transport (STT/TTS) with a `BusBridgeProcessor` where an LLM would normally go.
- **Voice/LLM agents** (`LLMAgent(bridged=())`): run their own LLM pipeline, receive frames from the bridge, transfer between each other.
- **Worker agents** (`BaseAgent`): receive tasks, process them, return results.

### Agent hierarchy

```
BaseAgent                    -- pipeline lifecycle, parent-child, tasks, activation
BaseAgent(bridged=())        -- adds edge processors for bus frame routing (all bridges)
BaseAgent(bridged=("voice",)) -- edge processors filtered to named bridges
  LLMAgent                   -- build_llm(), @tool registration, message injection on activation
  FlowsAgent                 -- Pipecat Flows integration (node-based conversation, always bridged)
```

### Key files

- `src/pipecat_subagents/agents/base_agent.py` -- BaseAgent, _BusEdgeProcessor, AgentActivationArgs, AgentReadyData, AgentErrorData
- `src/pipecat_subagents/agents/llm_agent.py` -- LLMAgent, LLMAgentActivationArgs
- `src/pipecat_subagents/agents/flows_agent.py` -- FlowsAgent
- `src/pipecat_subagents/agents/task_context.py` -- TaskContext, TaskGroup, TaskGroupContext, TaskGroupEvent, TaskGroupResponse, TaskGroupError, TaskStatus
- `src/pipecat_subagents/bus/bus.py` -- AgentBus abstract base
- `src/pipecat_subagents/bus/bridge_processor.py` -- BusBridgeProcessor (supports named bridges)
- `src/pipecat_subagents/bus/messages.py` -- All bus message types (BusFrameMessage has bridge field)
- `src/pipecat_subagents/registry/registry.py` -- AgentRegistry (async watch with immediate fire)
- `src/pipecat_subagents/runner/runner.py` -- AgentRunner

### Activation model

- `active` flag lives on `BaseAgent` (defaults to `False`)
- `activate_agent(name)` / `deactivate_agent(name)` send bus messages, handled by `BaseAgent`
- `on_activated(args)` / `on_deactivated()` hooks fire on the target agent
- `handoff_to(name)` on `BaseAgent` is a convenience: deactivates self locally, then activates target
- `activate_agent` and `handoff_to` accept `Optional[AgentActivationArgs]` (dataclass, not Pydantic)

### Registry

- Only root agents (added via `AgentRunner.add_agent()`) are announced to remote runners via the registry
- Child agents (added via `BaseAgent.add_agent()`) are only announced to their parent
- Use `watch_agent(name)` to receive `on_agent_ready` for any agent (works the same locally and distributed)
- `registry.watch()` is async and fires immediately if the agent is already registered
- Runner names must be unique across distributed setups (auto-generated with UUID by default)

### Bridge routing

- `BusBridgeProcessor(bridge="voice")` tags outgoing frames and filters incoming by bridge name
- `BaseAgent(bridged=("voice",))` only accepts frames from the "voice" bridge
- `BaseAgent(bridged=())` accepts frames from all bridges (default when bridged)
- `BusFrameMessage.bridge` field carries the bridge name (None for unnamed)
- Enables parallel pipelines (voice + video) or multiple agents on the same bridge

### Task lifecycle

- `request_task(*agent_names, payload=, timeout=)` sends work (fire-and-forget, callback-based)
- `task_group(*agent_names, payload=, timeout=)` returns a structured context manager
- Both wait for agents to be ready (via registry) before sending requests
- Workers receive `on_task_request(message)`, respond via `send_task_response()` or streaming
- `on_task_completed(result: TaskGroupResponse)` fires when all workers respond
- `on_task_error(message)` fires when a worker errors and cancel_on_error cancels the group
- Task completion does NOT end the agent's pipeline; agents stay alive for reuse

### Task group context

- `task_group()` returns `TaskGroupContext` (async context manager + async iterator)
- `async for event in tg` yields `TaskGroupEvent` (UPDATE, STREAM_START, STREAM_DATA, STREAM_END)
- `tg.responses` available after completion
- Raises `TaskGroupError` on timeout, worker error (with cancel_on_error), or ready-wait timeout
- Partial responses (including error response) available via `tg.responses` after catching error

### Task hooks (bus message pattern)

All task hooks receive the bus message directly (not individual arguments):
- `on_task_request(message: BusTaskRequestMessage)`
- `on_task_response(message: BusTaskResponseMessage)`
- `on_task_error(message: BusTaskResponseMessage)`
- `on_task_update(message: BusTaskUpdateMessage)`
- `on_task_update_requested(message: BusTaskUpdateRequestMessage)`
- `on_task_completed(result: TaskGroupResponse)`
- `on_task_stream_start/data/end(message: BusTaskStream*Message)`
- `on_task_cancelled(message: BusTaskCancelMessage)`

## Code conventions

- Google-style docstrings
- Docstrings explain purpose, not implementation. Don't describe which internal methods are called or how data flows internally. Do explain what developers need to know to use or extend the API.
- Don't enumerate specific message fields in hook docstrings; the type signature is sufficient
- No em dashes in docstrings or documentation. Use periods, colons, semicolons, or commas instead.
- Public methods: document with Args/Returns/Raises as needed
- Private methods (starting with `_`): don't add docstrings unless the logic is non-obvious
- Use backticks for code references in docstrings
- Lifecycle hooks should always call `super()` (e.g. `await super().on_activated(args)`)
- No Pydantic in agent layer; use dataclasses with `from_dict()`/`to_dict()` for serialization
