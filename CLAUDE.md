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
    LLMContextAgent          -- LLMAgent with built-in LLMContext + LLMContextAggregatorPair
  FlowsAgent                 -- Pipecat Flows integration (node-based conversation, always bridged)
```

### Key files

- `src/pipecat_subagents/agents/base_agent.py` -- BaseAgent, _BusEdgeProcessor, AgentActivationArgs, AgentReadyData, AgentErrorData
- `src/pipecat_subagents/agents/llm/llm_agent.py` -- LLMAgent, LLMAgentActivationArgs
- `src/pipecat_subagents/agents/llm/llm_context_agent.py` -- LLMContextAgent
- `src/pipecat_subagents/agents/llm/tool_decorator.py` -- @tool decorator
- `src/pipecat_subagents/agents/flows/flows_agent.py` -- FlowsAgent
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
- `BaseAgent.handoff_to` takes `activation_args=`. `LLMAgent.handoff_to` adds `messages=` (spoken before transfer) and `result_callback=`
- `LLMAgent.end` takes `messages=` (spoken before ending) and `result_callback=`
- `activate_agent` accepts `Optional[AgentActivationArgs]` (dataclass, not Pydantic)

### Registry

- Only root agents (added via `AgentRunner.add_agent()`) are announced to remote runners via the registry
- Child agents (added via `BaseAgent.add_agent()`) are private to the parent; lifecycle (end, cancel) is propagated automatically
- `add_agent()` does NOT auto-watch. To be notified when a child is ready, use `watch_agent(name)` or `@agent_ready(name="name")`
- `on_agent_ready` only fires for agents explicitly watched via `watch_agent(name)` or the `@agent_ready` decorator
- `@agent_ready(name="name")` is declarative sugar for `watch_agent`. Watches are usually registered after the agent is ready and activated, so handlers won't fire prematurely
- `watch_agent()` / `registry.watch()` fires immediately if the agent is already registered
- Runner names must be unique across distributed setups (auto-generated with UUID by default)

### Bridge routing

- `BusBridgeProcessor(bridge="voice")` tags outgoing frames and filters incoming by bridge name
- `BaseAgent(bridged=("voice",))` only accepts frames from the "voice" bridge
- `BaseAgent(bridged=())` accepts frames from all bridges (default when bridged)
- `BusFrameMessage.bridge` field carries the bridge name (None for unnamed)
- Enables parallel pipelines (voice + video) or multiple agents on the same bridge

### Task dispatch (requester side)

Two patterns for sending work to agents:

**Context managers (structured, recommended):**
- `task(agent_name, payload=, timeout=)` returns `TaskContext` for a single agent
- `task_group(*agent_names, payload=, timeout=)` returns `TaskGroupContext` for parallel dispatch
- Both are async context managers that wait for responses on exit
- Both support `async for event in ctx` to receive intermediate events (updates, streaming)
- `TaskContext` yields `TaskEvent`, `TaskGroupContext` yields `TaskGroupEvent` (includes `agent_name`)
- On error inside the `async with` block (including `CancelledError`), the task is automatically cancelled
- Cleanup is shielded so it completes even during tool interruption

**Fire-and-forget:**
- `request_task(agent_name, payload=, timeout=)` sends work, returns `task_id`
- `request_task_group(*agent_names, payload=, timeout=)` sends to multiple agents, returns `task_id`
- Use callbacks (`on_task_response`, `on_task_completed`) to handle results
- Caller must cancel manually if needed (e.g. on tool interruption)

Both patterns wait for agents to be ready (via registry) before sending requests.
Task completion does NOT end the agent's pipeline; agents stay alive for reuse.

### Task handling (worker side)

- Workers receive `on_task_request(message)` or use `@task` decorated handlers
- All `send_task_*` methods require an explicit `task_id` argument (from `message.task_id`)
- `send_task_response(task_id, response, status=)` sends result and removes the task from `active_tasks`
- `send_task_update(task_id, update)` sends progress without completing
- `send_task_stream_start/data/end(task_id, data)` for streaming results
- `active_tasks` property returns `dict[str, BusTaskRequestMessage]` of in-flight tasks
- Task handlers always run in their own asyncio task so the bus message loop is never blocked
- Multiple tasks can be in flight simultaneously
- When the agent stops, any still-active tasks are automatically reported as `CANCELLED`

### Task cancellation

- `cancel_task(task_id, reason=)` sends `BusTaskCancelMessage` to all agents in the group
- Worker receives `on_task_cancelled(message)`, then auto-sends a `CANCELLED` response
- Context managers (`task()`, `task_group()`) cancel automatically on exception or `CancelledError`
- For fire-and-forget tasks, the user must cancel manually. Pattern for tool interruption:

```python
task_ids = []
try:
    task_ids.append(await self.request_task("w1", payload=data))
    task_ids.append(await self.request_task("w2", payload=data))
    # ... do work ...
except asyncio.CancelledError:
    for tid in task_ids:
        await self.cancel_task(tid, reason="tool cancelled")
    raise
```

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
