# Source-to-Doc Mapping

Maps pipecat-subagents source files to their documentation pages. Source paths are relative to `src/pipecat_subagents/`. Doc paths are relative to `DOCS_PATH`.

## Direct mapping

| Source path | Primary doc page | Also check guides |
|---|---|---|
| `agents/base_agent.py` | `api-reference/pipecat-subagents/base-agent.mdx` | `subagents/learn/agents-and-runner.mdx`, `subagents/learn/agent-handoff.mdx`, `subagents/learn/task-coordination.mdx` |
| `agents/llm/llm_agent.py` | `api-reference/pipecat-subagents/llm-agent.mdx` | `subagents/learn/llm-agent-and-tool-decorator.mdx`, `subagents/learn/agent-handoff.mdx`, `subagents/learn/custom-voices-per-agent.mdx` |
| `agents/flows/flows_agent.py` | `api-reference/pipecat-subagents/flows-agent.mdx` | `subagents/learn/structured-conversations-with-flows-agent.mdx` |
| `agents/task_context.py` | `api-reference/pipecat-subagents/types.mdx` | `subagents/learn/task-coordination.mdx` |
| `agents/llm/tool_decorator.py` | `api-reference/pipecat-subagents/decorators.mdx` (@tool) | `subagents/learn/llm-agent-and-tool-decorator.mdx` |
| `agents/task_decorator.py` | `api-reference/pipecat-subagents/decorators.mdx` (@task) | `subagents/learn/task-coordination.mdx` |
| `agents/watch_decorator.py` | `api-reference/pipecat-subagents/decorators.mdx` (@agent_ready) | `subagents/fundamentals/agent-registry-and-discovery.mdx` |
| `bus/bus.py` | `api-reference/pipecat-subagents/bus.mdx` (AgentBus) | `subagents/fundamentals/agent-bus.mdx` |
| `bus/messages.py` | `api-reference/pipecat-subagents/messages.mdx` | -- |
| `bus/bridge_processor.py` | `api-reference/pipecat-subagents/bus.mdx` (BusBridgeProcessor) | `subagents/fundamentals/understanding-the-bus-bridge.mdx` |
| `bus/subscriber.py` | `api-reference/pipecat-subagents/bus.mdx` (BusSubscriber) | -- |
| `bus/local/async_queue.py` | `api-reference/pipecat-subagents/bus.mdx` (AsyncQueueBus) | `subagents/fundamentals/agent-bus.mdx` |
| `bus/network/redis.py` | `api-reference/pipecat-subagents/bus.mdx` (RedisBus) | `subagents/fundamentals/agent-bus.mdx`, `subagents/learn/distributed-agents.mdx` |
| `bus/serializers/base.py` | `api-reference/pipecat-subagents/serializers.mdx` | -- |
| `bus/serializers/json.py` | `api-reference/pipecat-subagents/serializers.mdx` | -- |
| `bus/adapters/*.py` | `api-reference/pipecat-subagents/serializers.mdx` | -- |
| `runner/runner.py` | `api-reference/pipecat-subagents/agent-runner.mdx` | `subagents/learn/agents-and-runner.mdx` |
| `registry/registry.py` | `api-reference/pipecat-subagents/types.mdx` (AgentRegistry) | `subagents/fundamentals/agent-registry-and-discovery.mdx` |
| `types.py` | `api-reference/pipecat-subagents/types.mdx` | -- |
| `agents/proxy/websocket/client.py` | `api-reference/pipecat-subagents/proxy-agents.mdx` | `subagents/learn/proxy-agents.mdx` |
| `agents/proxy/websocket/server.py` | `api-reference/pipecat-subagents/proxy-agents.mdx` | `subagents/learn/proxy-agents.mdx` |

## Skip list

These files should never trigger doc updates.

| Pattern | Reason |
|---|---|
| `**/__init__.py` | Re-exports only |
| `clowder/**` | Observability tool, no docs coverage |
| `bus/queue.py` | Internal priority queue |

## Search fallback

For files not in the tables above:
1. Extract the main class name(s) from the source file
2. Search the docs directory for that class name: `grep -r "ClassName" DOCS_PATH/api-reference/pipecat-subagents/` and `grep -r "ClassName" DOCS_PATH/subagents/`
3. If found in a doc page, use that as the mapping
4. If not found, the file is **unmapped** -- report it in the summary
