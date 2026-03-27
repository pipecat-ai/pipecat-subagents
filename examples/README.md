# Pipecat Subagents Examples

This directory contains example implementations demonstrating the Pipecat Subagents framework.

## Local

Examples where all agents run in the same process.

- **[`local/agent-handoff/`](local/agent-handoff/)**: Agents that transfer control between each other during a conversation.
- **[`local/parallel-debate/`](local/parallel-debate/)**: Parallel task workers that debate a topic from multiple perspectives using a task group context.
- **[`local/code-assistant/`](local/code-assistant/)**: Voice code assistant that explores your codebase using Claude Agent SDK.

## Distributed

Examples where agents run across separate processes or machines.

- **[`distributed/redis-handoff/`](distributed/redis-handoff/)**: Agents on separate machines connected via Redis pub/sub.
- **[`distributed/remote-proxy-assistant/`](distributed/remote-proxy-assistant/)**: LLM agent on a remote server connected via WebSocket proxy.
