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
- **Voice/LLM agents** (`LLMDetachedAgent`): run their own LLM pipeline, receive frames from the bridge, transfer between each other.
- **Worker agents** (`BaseAgent`): receive tasks, process them, return results.

### Agent hierarchy

```
BaseAgent                 -- pipeline lifecycle, parent-child, tasks, activation
  DetachedAgent           -- bus frame routing via edge processors, handoff_to()
    LLMDetachedAgent      -- build_llm(), @tool registration, message injection on activation
    FlowsDetachedAgent    -- Pipecat Flows integration (node-based conversation)
```

### Key files

- `src/pipecat_subagents/agents/base_agent.py` -- BaseAgent
- `src/pipecat_subagents/agents/detached_agent.py` -- DetachedAgent + `_BusEdgeProcessor`
- `src/pipecat_subagents/agents/llm_detached_agent.py` -- LLMDetachedAgent
- `src/pipecat_subagents/agents/flows_detached_agent.py` -- FlowsDetachedAgent
- `src/pipecat_subagents/bus/bus.py` -- AgentBus abstract base
- `src/pipecat_subagents/bus/bridge_processor.py` -- BusBridgeProcessor
- `src/pipecat_subagents/bus/messages.py` -- All bus message types
- `src/pipecat_subagents/registry/registry.py` -- AgentRegistry
- `src/pipecat_subagents/runner/runner.py` -- AgentRunner

### Activation model

- `active` flag lives on `BaseAgent` (defaults to `True`; `DetachedAgent` overrides to `False`)
- `activate_agent(name)` / `deactivate_agent(name)` send bus messages, handled by `BaseAgent`
- `on_activated(args)` / `on_deactivated()` hooks fire on the target agent
- `handoff_to(name)` on `DetachedAgent` is a convenience: deactivates self locally, then activates target
- `active=True` in DetachedAgent constructor just sets the flag; no `on_activated` fires

### Registry

- Only root agents (added via `AgentRunner.add_agent()`) are broadcast to other local agents and remote runners
- Child agents (added via `BaseAgent.add_agent()`) are only announced to their parent
- Other agents can opt in to child notifications via `watch_agent(name)`
- Runner names must be unique across distributed setups (auto-generated with UUID by default)

### Task lifecycle

- Parent calls `start_task(*agents, payload=, timeout=)` to launch workers
- Workers receive `on_task_request()`, respond via `send_task_response()` or streaming
- `on_task_completed()` fires on the parent when all workers in a group have responded
- Task completion does NOT end the agent's pipeline; agents stay alive for reuse

## Code conventions

- Google-style docstrings
- Docstrings explain purpose, not implementation. Don't describe which internal methods are called or how data flows internally. Do explain what developers need to know to use or extend the API.
- No em dashes in docstrings or documentation. Use periods, colons, semicolons, or commas instead.
- Public methods: document with Args/Returns/Raises as needed
- Private methods (starting with `_`): don't add docstrings unless the logic is non-obvious
- Use backticks for code references in docstrings
- Lifecycle hooks should always call `super()` (e.g. `await super().on_activated(args)`)
