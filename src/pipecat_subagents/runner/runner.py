#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent runner for orchestrating multi-agent lifecycle and pipeline tasks."""

import asyncio
import uuid
from typing import Optional

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.utils.base_object import BaseObject

from pipecat_subagents.agents.base_agent import BaseAgent
from pipecat_subagents.bus import (
    AgentBus,
    AsyncQueueBus,
    BusAddAgentMessage,
    BusAgentRegistryMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusMessage,
)
from pipecat_subagents.bus.subscriber import BusSubscriber
from pipecat_subagents.registry import AgentRegistry
from pipecat_subagents.types import RegisteredAgentData


class AgentRunner(BaseObject, BusSubscriber):
    """Lifecycle orchestrator for multi-agent systems.

    Manages agent pipelines, coordinates startup and shutdown, and
    responds to bus messages. On graceful end, root agents are ended
    first; parent agents propagate shutdown to their children.

    Event handlers:

    on_runner_started(runner)
        Fired after all registered agents have been started.

    Example::

        runner = AgentRunner()

        agent = MyAgent("my_agent", bus=runner.bus, ...)
        await runner.add_agent(agent)

        await runner.run()
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        bus: Optional[AgentBus] = None,
        handle_sigint: bool = True,
    ):
        """Initialize the `AgentRunner`.

        Args:
            name: Optional unique name for this runner. Defaults to a
                UUID-based name. Must be unique across all runners in a
                distributed setup.
            bus: Optional `AgentBus` instance. Creates an `AsyncQueueBus`
                if not provided.
            handle_sigint: Whether to handle SIGINT for graceful shutdown.
                Defaults to True.
        """
        super().__init__(name=name or f"runner-{uuid.uuid4().hex[:8]}")
        self._bus = bus or AsyncQueueBus()
        self._registry = AgentRegistry(runner_name=self.name)

        self._running: bool = False
        self._agents: dict[str, BaseAgent] = {}
        self._running_agent_tasks: dict[str, asyncio.Task] = {}
        self._pipecat_runner = PipelineRunner(handle_sigint=handle_sigint)
        self._shutdown_event = asyncio.Event()
        self._known_runners: set[str] = set()

        self._register_event_handler("on_runner_started")

    @property
    def bus(self) -> AgentBus:
        """The bus instance for agent communication."""
        return self._bus

    @property
    def registry(self) -> AgentRegistry:
        """The shared agent registry."""
        return self._registry

    async def on_bus_message(self, message: BusMessage) -> None:
        """Handle bus messages directed at the runner.

        Args:
            message: The bus message to handle.
        """
        if message.source == self.name:
            return
        if isinstance(message, BusEndMessage):
            asyncio.create_task(self.end(message.reason))
        elif isinstance(message, BusCancelMessage):
            asyncio.create_task(self.cancel(message.reason))
        elif isinstance(message, BusAddAgentMessage) and message.agent:
            await self.add_agent(message.agent)
        elif isinstance(message, BusAgentRegistryMessage):
            await self._handle_agent_registry(message)

    async def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to this runner.

        Can be called before or after run(). When called after run() has
        started, the agent's pipeline task is created and started immediately.

        Args:
            agent: The agent to add.

        Raises:
            ValueError: If an agent with this name already exists.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already exists")
        agent.set_registry(self._registry)
        self._registry.watch(agent.name, self._on_agent_ready)
        self._agents[agent.name] = agent
        logger.debug(f"AgentRunner '{self}': added agent '{agent.name}'")

        if self._running:
            await self._start_agent_task(agent)

    async def run(self) -> None:
        """Start all agents, block until shutdown.

        Starts all registered agents, fires ``on_runner_started``, then blocks
        until `end()` or `cancel()` is called. New agents can be added
        dynamically via `add_agent()` after ``run()`` has started.
        """
        self._running = True
        self._shutdown_event.clear()

        await self._bus.subscribe(self)
        await self._bus.start()

        for agent in self._agents.values():
            await self._start_agent_task(agent)

        await self._call_event_handler("on_runner_started")

        await self._shutdown_event.wait()

        # Wait for remaining agent tasks to finish cleanup
        remaining = [t for t in self._running_agent_tasks.values() if not t.done()]
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)

        await self._bus.stop()
        self._running = False

    async def end(self, reason: Optional[str] = None) -> None:
        """Gracefully end all agents and shut down.

        Ends root agents first; parent agents propagate shutdown to
        their children automatically. Idempotent; subsequent calls
        are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"AgentRunner '{self}': ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name, agent in self._agents.items():
            if agent.parent is None:
                await self._bus.send(
                    BusEndAgentMessage(source=self.name, target=name, reason=reason)
                )

    async def cancel(self, reason: Optional[str] = None) -> None:
        """Immediately cancel all agents and shut down.

        Cancels root agents first; parent agents propagate cancellation
        to their children automatically. Idempotent.subsequent calls
        are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"AgentRunner '{self}': cancelling (reason={reason})")
        self._shutdown_event.set()
        for name, agent in self._agents.items():
            if agent.parent is None:
                await self._bus.send(
                    BusCancelAgentMessage(source=self.name, target=name, reason=reason)
                )
        await self._pipecat_runner.cancel()

    async def _start_agent_task(self, agent: BaseAgent) -> None:
        """Create an agent's pipeline task and start it as a background asyncio task."""
        logger.debug(f"AgentRunner '{self}': starting agent '{agent.name}'")
        try:
            pipeline_task = await agent.create_pipeline_task()
        except Exception:
            logger.exception(
                f"AgentRunner '{self}': failed to create pipeline task for agent '{agent.name}'"
            )
            return

        asyncio_task = asyncio.create_task(
            self._pipecat_runner.run(pipeline_task),
            name=f"agent_{agent.name}",
        )

        # Register the agent task.
        self._running_agent_tasks[agent.name] = asyncio_task
        asyncio_task.add_done_callback(self._on_agent_task_done)

        # Add the task to event loop right away without needing to `await`.
        await asyncio.sleep(0)

    def _on_agent_task_done(self, task: asyncio.Task) -> None:
        """Remove a completed agent task."""
        name = task.get_name().removeprefix("agent_")
        self._running_agent_tasks.pop(name, None)

    async def _on_agent_ready(self, agent_data: RegisteredAgentData) -> None:
        """Called when a local agent registers as ready.

        For root agents, broadcasts ``on_agent_ready`` to all local
        agents and sends an updated registry to remote runners.
        Child readiness is handled by the parent directly (via its
        own watch).
        """
        if agent_data.runner != self.name:
            return

        agent = self._agents.get(agent_data.agent_name)
        if not agent or agent.parent is not None:
            # Child agent: parent handles it via its own watch
            return

        # Root agent: broadcast to all local agents
        for other in self._agents.values():
            if other.name != agent_data.agent_name:
                await other._on_watched_agent_ready(agent_data)

        await self._send_registry()

    async def _send_registry(self) -> None:
        """Broadcast this runner's root agents to the bus."""
        agents = [name for name, agent in self._agents.items() if agent.parent is None]
        if agents:
            logger.debug(f"AgentRunner '{self}': broadcasting registry: {agents}")
            await self._bus.send(
                BusAgentRegistryMessage(
                    source=self.name,
                    runner=self.name,
                    agents=agents,
                )
            )

    async def _handle_agent_registry(self, message: BusAgentRegistryMessage) -> None:
        """Handle a registry from a remote runner.

        Merges remote agents into the shared registry and, if this runner
        hasn't seen this remote runner before, sends its own registry back
        so both sides discover each other.
        """
        logger.debug(
            f"AgentRunner '{self}': received registry from '{message.runner}' with agents: {message.agents}"
        )
        for agent_name in message.agents:
            await self._registry.register(
                RegisteredAgentData(agent_name=agent_name, runner=message.runner)
            )
        if message.runner not in self._known_runners:
            self._known_runners.add(message.runner)
            logger.debug(
                f"AgentRunner '{self}': new runner '{message.runner}', sending our registry back"
            )
            await self._send_registry()
