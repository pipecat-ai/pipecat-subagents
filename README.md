<h1><div align="center">
 <img alt="pipecat subagents" width="500px" height="auto" src="https://github.com/pipecat-ai/pipecat-subagents/raw/refs/heads/main/pipecat-subagents.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-subagents)](https://pypi.org/project/pipecat-ai-subagents) ![Tests](https://github.com/pipecat-ai/pipecat-subagents/actions/workflows/tests.yaml/badge.svg) [![codecov](https://codecov.io/gh/pipecat-ai/pipecat-subagents/graph/badge.svg?token=LNVUIVO4Y9)](https://codecov.io/gh/pipecat-ai/pipecat-subagents) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/subagents/introduction) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Subagents is a distributed multi-agent framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme). Each agent runs its own Pipecat pipeline and communicates with other agents through a shared message bus, enabling you to decompose complex systems into specialized, coordinating agents that can run locally or across machines.

Whether local or distributed, the programming model is the same: create an `AgentRunner`, connect it to the bus, and add agents.

## ✨ Features

- **Per-agent pipeline**: every agent runs its own Pipecat pipeline with access to all services, processors, and transports (e.g. different LLMs, local models, image analysis, custom processors).
- **Agent handoff**: transfer control between agents mid-conversation, seamlessly switching context and capabilities.
- **Task coordination**: dispatch work to specialized worker agents in parallel, track progress with timeouts and cancellation, and stream results back.
- **Distributed agents**: run agents across machines, scaling each independently. Agents communicate via a shared bus that works the same locally or over the network.
- **Mixed agent types**: combine free-form LLM agents with structured [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) agents or wrap third-party frameworks as subagents.
- **Agent lifecycle**: activation, deactivation, error propagation, and graceful shutdown across a tree of agents.

## 🧠 Why Pipecat Subagents?

Pipecat is a powerful real-time framework for building voice and multimodal AI applications. Pipecat Subagents extends it with distributed multi-agent coordination where each agent runs its own Pipecat pipeline.

**Some things you can build:**

- A customer support system where each specialist runs its own LLM with dedicated tools and context, transferring seamlessly between each other.
- A video game where multiple LLMs independently control different characters, environments, or game mechanics.
- A stock analysis app that dispatches parallel research to multiple worker agents and synthesizes their findings.
- A video or image analysis pipeline where worker agents process media using Pipecat processors and stream updates back.
- An IoT system where remote devices run agents on specialized hardware, reporting status and receiving commands.

If a single Pipecat pipeline covers your use case, you don't need subagents. When you outgrow it, the transition is straightforward: your existing pipeline becomes one agent among many.

## 📦 Installation

```bash
uv add pipecat-ai-subagents

# or: pip install pipecat-ai-subagents
```

> Requires Python 3.10+ and [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started).

## 🚀 Examples

See the [examples](examples/) directory for complete, runnable demos.

## 🔍 Observability with Clowder

Clowder (a group of cats) is a real-time observability tool for Pipecat Subagents. It shows runners, agents, tasks, and bus messages in a web UI, whether local or distributed.

<p align="center"><img src="https://github.com/pipecat-ai/pipecat-subagents/raw/refs/heads/main/clowder-image.png" alt="Clowder" width="500"/></p>

### Adding Clowder to your system

Add a `ClowderAgent` to any runner:

```python
from pipecat_subagents.clowder import ClowderAgent

clowder = ClowderAgent("clowder", bus=runner.bus, port=7070)
await runner.add_agent(clowder)
```

Or use a setup file to add Clowder without modifying your application code. Create a file (e.g. `clowder_setup.py`):

```python
from pipecat_subagents.clowder import ClowderAgent
from pipecat_subagents.runner import AgentRunner


async def setup_runner(runner: AgentRunner):
    clowder = ClowderAgent(bus=runner.bus)
    await runner.add_agent(clowder)
```

Then set the environment variable:

```bash
export PIPECAT_SUBAGENTS_SETUP_FILES=/path/to/clowder_setup.py
```

### Running the UI

```bash
cd clowder-ui
npm install
npm run dev
```

Open http://localhost:5173 and click **Connect**. The UI shows agents grouped by runner, tasks with live duration, and a filterable bus message stream. Connect at any time; all messages are buffered.

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat-subagents/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## 💬 Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Reach us on X](https://x.com/pipecat_ai)
