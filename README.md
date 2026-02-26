<h1><div align="center">
 <img alt="pipecat agents" width="500px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat-agents/main/pipecat-agents.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-agents)](https://pypi.org/project/pipecat-ai-agents) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/guides/features/pipecat-agents) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Agents is a distributed multi-agent framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme). Each agent runs its own Pipecat pipeline and communicates with other agents through a shared message bus, enabling you to decompose complex systems into specialized, coordinating agents that can run locally or across machines.

Because each agent is just a Pipecat pipeline, anything you can build with Pipecat works as an agent:

- A voice assistant that appears as one agent but is actually multiple specialized agents behind the scenes
- A coordinator that spawns long-running background agents and collects results through the bus
- Agents running on different machines or specialized hardware, communicating over the same bus

Whether local or distributed, the programming model is the same: create an `AgentRunner`, connect it to the bus, and add agents.

## Dependencies

- Python 3.10 or higher
- [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started)

## Installation

1. Install uv

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   > **Need help?** Refer to the [uv install documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. Install the module

   ```bash
   # For new projects
   uv init my-pipecat-agents-app
   cd my-pipecat-agents-app
   uv add pipecat-ai-agents

   # Or for existing projects
   uv add pipecat-ai-agents
   ```

> **Using pip?** You can still use `pip install pipecat-ai-agents` to get set up.

## Key Concepts

### Agents

Agents are the core building blocks. Each agent runs its own Pipecat pipeline and communicates with other agents via the bus.

- **`BaseAgent`** — Abstract base class for all agents. Handles bus subscription, pipeline lifecycle, and agent transfer.
- **`LLMAgent`** — Agent with an LLM pipeline. Supports tool registration, message injection on activation, and function call result handling.
- **`FlowsAgent`** — Agent that integrates [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) for structured, node-based conversation logic.

### Bus

The message bus provides pub/sub communication between agents and the runner.

- **`AgentBus`** — Abstract base for inter-agent messaging.
- **`LocalAgentBus`** — In-process implementation backed by `asyncio.Queue`.
- **`BusBridgeProcessor`** — Bidirectional bridge that sends pipeline frames to other agents through the bus, and receives frames back into the pipeline.
- **`BusOutputProcessor`** — One-way bridge that captures pipeline output and publishes it to the bus.

### Runner

- **`AgentRunner`** — Orchestrates agent lifecycle, creates pipeline tasks, and coordinates shutdown. Agents can be added dynamically at runtime.

## Examples

The [examples](examples/) directory includes complete working implementations:

- **[single_agent.py](examples/single_agent.py)** — Simplest usage: one agent running a full voice pipeline through the AgentRunner
- **[two_llm_agents.py](examples/two_llm_agents.py)** — Two LLM agents (greeter + support) that transfer control between each other
- **[llm_and_flows_agent.py](examples/llm_and_flows_agent.py)** — Mixing agent types: an LLM agent and a Flows agent with structured conversation nodes

See the [examples README](examples/README.md) for setup and running instructions.

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

➡️ [Pipecat Agents Guide](https://docs.pipecat.ai/guides/pipecat-agents)

➡️ [Reach us on X](https://x.com/pipecat_ai)
