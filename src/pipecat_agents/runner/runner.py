#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Agent runner for orchestrating multi-agent lifecycle and pipeline tasks."""

import asyncio
from typing import Optional

from loguru import logger
from pipecat.pipeline.runner import PipelineRunner
from pipecat.utils.base_object import BaseObject

from pipecat_agents.agents.base_agent import BaseAgent
from pipecat_agents.bus import (
    AgentBus,
    BusAddAgentMessage,
    BusCancelAgentMessage,
    BusCancelMessage,
    BusEndAgentMessage,
    BusEndMessage,
    BusMessage,
    LocalAgentBus,
)
from pipecat_agents.bus.subscriber import BusSubscriber


class AgentRunner(BaseObject, BusSubscriber):
    """Lifecycle orchestrator for multi-agent systems.

    Manages agent pipelines, coordinates startup and shutdown, and
    responds to bus messages. On graceful end, root agents are ended
    first — parent agents propagate shutdown to their children.

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
        bus: Optional[AgentBus] = None,
        handle_sigint: bool = True,
    ):
        """Initialize the `AgentRunner`.

        Args:
            bus: Optional `AgentBus` instance. Creates a `LocalAgentBus`
                if not provided.
            handle_sigint: Whether to handle SIGINT for graceful shutdown.
                Defaults to True.
        """
        super().__init__()
        self._bus = bus or LocalAgentBus()

        self._running: bool = False
        self._agents: dict[str, BaseAgent] = {}
        self._running_agent_tasks: dict[str, asyncio.Task] = {}
        self._pipecat_runner = PipelineRunner(handle_sigint=handle_sigint)
        self._shutdown_event = asyncio.Event()

        self._register_event_handler("on_runner_started")

    @property
    def bus(self) -> AgentBus:
        """The bus instance for agent communication."""
        return self._bus

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
        self._agents[agent.name] = agent
        logger.debug(f"{self}: added agent '{agent.name}'")

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
        their children automatically. Idempotent — subsequent calls
        are ignored.

        Args:
            reason: Optional human-readable reason for ending.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"{self}: ending gracefully (reason={reason})")
        self._shutdown_event.set()
        for name, agent in self._agents.items():
            if agent.parent is None:
                await self._bus.send(
                    BusEndAgentMessage(source=self.name, target=name, reason=reason)
                )

    async def cancel(self, reason: Optional[str] = None) -> None:
        """Immediately cancel all agents and shut down.

        Cancels root agents first; parent agents propagate cancellation
        to their children automatically. Idempotent — subsequent calls
        are ignored.

        Args:
            reason: Optional human-readable reason for cancelling.
        """
        if self._shutdown_event.is_set():
            return
        logger.debug(f"{self}: cancelling (reason={reason})")
        self._shutdown_event.set()
        for name, agent in self._agents.items():
            if agent.parent is None:
                await self._bus.send(
                    BusCancelAgentMessage(source=self.name, target=name, reason=reason)
                )
        await self._pipecat_runner.cancel()

    async def _start_agent_task(self, agent: BaseAgent) -> None:
        """Create an agent's pipeline task and start it as a background asyncio task."""
        logger.debug(f"{self}: starting agent '{agent.name}'")
        try:
            pipeline_task = await agent.create_pipeline_task()
        except Exception:
            logger.exception(f"{self}: failed to create pipeline task for agent '{agent.name}'")
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
        """Remove a completed agent task and signal its finished event."""
        name = task.get_name().removeprefix("agent_")
        self._running_agent_tasks.pop(name, None)
        agent = self._agents.get(name)
        if agent:
            agent.notify_finished()
